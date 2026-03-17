import math
import unittest
from itertools import product

import mlx.core as mx
import mlx_tests
import numpy as np


def mlx_ref_attn(q, k, v, scale=1.0, mask=None, sinks=None):
    q_dtype = q.dtype
    q = q * mx.array(scale, q_dtype)
    n_q_heads = q.shape[-3]
    n_kv_heads = k.shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    B = q.shape[0]
    L = q.shape[2]
    kL = k.shape[2]

    if n_repeats > 1:
        q = mx.reshape(q, [B, n_kv_heads, n_repeats, L, -1])
        k = mx.expand_dims(k, 2)
        v = mx.expand_dims(v, 2)

    scores = q @ mx.swapaxes(k, -1, -2)
    is_causal = mask == "causal"
    if mask is not None:

        if is_causal:
            offset = kL - L
            q_indices = mx.arange(L) + offset
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]

        if n_repeats > 1 and mask.ndim >= 3:
            if mask.shape[-3] == 1:
                mask = mx.expand_dims(mask, -3)
            else:
                mask = mx.unflatten(mask, -3, (n_kv_heads, n_repeats))

        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores += mask

    if sinks is not None:
        sinks = mx.expand_dims(sinks, (0, 2, 3))
        if n_repeats > 1:
            sinks = mx.unflatten(sinks, 1, (n_kv_heads, n_repeats))
        score_shape = list(scores.shape)
        score_shape[-1] = 1
        sinks = mx.broadcast_to(sinks, score_shape)
        scores = mx.concatenate([sinks, scores], axis=-1)

    scores = mx.softmax(scores, axis=-1, precise=True)
    if sinks is not None:
        scores = scores[..., 1:]

    out = scores @ v
    if n_repeats > 1:
        out = mx.reshape(out, [B, n_q_heads, L, -1])
    return out


def do_attention(f, q, k, v, scale, mask=None, transpose=False):
    if transpose:
        q_t = mx.transpose(q, (0, 2, 1, 3))
        k_t = mx.transpose(k, (0, 2, 1, 3))
        v_t = mx.transpose(v, (0, 2, 1, 3))
        o_t = f(q_t, k_t, v_t, scale=scale, mask=mask)
        return mx.transpose(o_t, (0, 2, 1, 3))
    else:
        return f(q, k, v, scale=scale, mask=mask)


def prepare_inputs(B, qL, kL, D, qH, kH, mask, transpose, dtype):
    mx.random.seed(0)

    scale = 1.0 / math.sqrt(D)
    shape_q = (B, qL, qH, D) if transpose else (B, qH, qL, D)
    shape_kv = (B, kL, kH, D) if transpose else (B, kH, kL, D)

    q = mx.random.uniform(0.0, 0.5, shape_q, dtype)
    k = mx.random.uniform(0.0, 0.5, shape_kv, dtype)
    v = mx.random.uniform(0.0, scale, shape_kv, dtype)

    if mask is not None:
        if mask == "additive":
            mask = mx.random.uniform(0.0, 0.5, (B, qH, qL, kL), dtype)
        elif mask == "bool":
            mask = mx.random.uniform(0.0, 1.0, (B, qH, qL, kL)) < 0.5

    return q, k, v, scale, mask


# SDPA for MHA (n_heads == n_kv_heads)
def mlx_primitives_sdpa(q, k, v, scale, mask=None):
    p = (q * scale) @ k.transpose(0, 1, 3, 2)
    qL = q.shape[2]
    kL = k.shape[2]
    is_causal = mask == "causal"
    if mask is not None:
        if is_causal:
            offset = kL - qL
            q_indices = mx.arange(qL) + offset
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]
            p = mx.where(mask, p, mx.finfo(mx.float32).min)
        elif mask.dtype == mx.bool_:
            p = mx.where(mask, p, mx.finfo(mx.float32).min)
        else:
            p += mask
    scores = mx.softmax(p.astype(mx.float32), axis=-1).astype(p.dtype)
    return scores @ v


class TestFastSDPA(mlx_tests.MLXTestCase):
    def test_sdpa_vector_kv_transposed_head_seq(self):
        D = 64
        Nq = 4
        Nkv = 1
        scale = 1.0
        mx.random.seed(0)
        q = 5e-1 * mx.random.normal(shape=(1, Nq, 1, D))

        lengths = [43, 4096]
        for L in lengths:
            k = 5e-1 * mx.random.normal(shape=(1, L, Nkv, D))
            v = 5e-1 * mx.random.normal(shape=(1, L, Nkv, D))
            k = k.swapaxes(1, 2)
            v = v.swapaxes(1, 2)
            masks = [
                mx.array(True),
                mx.array([True] * (L - 10) + [False] * 10),
                mx.random.uniform(shape=(Nq, 1, L)) > 0.2,
                mx.random.uniform(shape=(L, 1, Nq)).T > 0.2,
            ]

            for m in masks:
                ref = mlx_primitives_sdpa(q, k, v, scale, mask=m)
                out = mx.fast.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    scale=scale,
                    mask=m,
                )
                self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_sdpa_vector(self):
        D = 64
        L = 43
        Nq = 4
        Nkv = 1
        scale = 1.0
        mx.random.seed(0)
        q = 5e-1 * mx.random.normal(shape=(1, Nq, 1, D))
        k = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        v = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))

        with self.assertRaises(ValueError):
            mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                mask=mx.full((Nq, 2, L), False),
            )

        masks = [
            None,
            mx.array(True),
            mx.array([True] * (L - 10) + [False] * 10),
            mx.random.uniform(shape=(Nq, 1, L)) > 0.2,
            mx.random.uniform(shape=(L, 1, Nq)).T > 0.2,
            mx.random.uniform(shape=(Nq, 1, L)),
            mx.random.uniform(shape=(L, 1, Nq)).T,
            mx.log(mx.random.uniform(shape=(Nq, 1, L)) > 0.2),
            mx.log(mx.random.uniform(shape=(L, 1, Nq)).T > 0.2),
            "causal",
        ]
        for m in masks:
            ref = mlx_primitives_sdpa(q, k, v, scale, mask=m)
            out = mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                mask=m,
            )
            self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

        L = 4096
        scale = 1.0
        mx.random.seed(0)
        q = 5e-1 * mx.random.normal(shape=(1, Nq, 1, D))
        k = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        v = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))

        masks = [
            mx.array(True),
            mx.array([True] * (L - 10) + [False] * 10),
            mx.random.uniform(shape=(Nq, 1, L)) > 0.2,
            mx.random.uniform(shape=(L, 1, Nq)).T > 0.2,
            mx.random.uniform(shape=(Nq, 1, L)),
            mx.random.uniform(shape=(L, 1, Nq)).T,
            mx.log(mx.random.uniform(shape=(Nq, 1, L)) > 0.2),
            mx.log(mx.random.uniform(shape=(L, 1, Nq)).T > 0.2),
            "causal",
        ]
        for m in masks:
            ref = mlx_primitives_sdpa(q, k, v, scale, mask=m)
            out = mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                mask=m,
            )
            self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_sdpa_fully_masked(self):
        Lkv = 8
        mask = mx.array(False)
        for D in [128]:
            for Lq in [1, 8, 32]:
                q = mx.random.normal(shape=(1, 4, Lq, D))
                k = mx.random.normal(shape=(1, 4, Lkv, D))
                v = mx.random.normal(shape=(1, 4, Lkv, D))

                out = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=1)
                self.assertFalse(mx.any(mx.isnan(out)))

    def test_sdpa_inf_score(self):
        Lkv = 8
        for D in [4, 128]:
            for Lq in [1, 8]:
                q = mx.ones(shape=(1, 4, Lq, D))
                k = mx.ones(shape=(1, 4, Lkv, D))
                v = mx.random.normal(shape=(1, 4, Lkv, D))
                k[..., 0, :] = -float("inf")
                ref = mlx_primitives_sdpa(q, k, v, scale=1, mask=None)
                out = mx.fast.scaled_dot_product_attention(q, k, v, mask=None, scale=1)
                self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_sdpa_few_query(self):
        D = 64
        L = 43
        Lq = 8
        Nq = 8
        Nkv = 1
        scale = 1.0
        mx.random.seed(0)
        q = 5e-1 * mx.random.normal(shape=(1, Lq, Nq, D))
        q = q.swapaxes(1, 2)
        k = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        v = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))

        masks = [
            None,
            mx.array(True),
            mx.array([True] * (L - 10) + [False] * 10),
            mx.random.uniform(shape=(Nq, 1, L)) > 0.2,
            mx.random.uniform(shape=(L, 1, Nq)).T > 0.2,
            "causal",
        ]
        for m in masks:
            ref = mlx_primitives_sdpa(q, k, v, scale, mask=m)
            out = mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                mask=m,
            )
            self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

        L = 4096
        scale = 1.0
        mx.random.seed(0)
        q = 5e-1 * mx.random.normal(shape=(1, Nq, Lq, D))
        k = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        v = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))

        masks = [
            None,
            mx.array(True),
            mx.array([True] * (L - 10) + [False] * 10),
            mx.random.uniform(shape=(Nq, 1, L)) > 0.2,
            mx.random.uniform(shape=(L, 1, Nq)).T > 0.2,
            "causal",
        ]
        for m in masks:
            ref = mlx_primitives_sdpa(q, k, v, scale, mask=m)
            out = mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                mask=m,
            )
            self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    @unittest.skip("Different head and value dims is not enabled")
    def test_sdpa_vector_value_dims(self):
        D = 192
        V = 128
        Nq = 4
        Nkv = 1
        scale = 1.0
        mx.random.seed(0)

        for L in [43, 128, 237, 8192]:
            q = 5e-1 * mx.random.normal(shape=(1, Nq, 1, D))
            k = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
            v = 5e-1 * mx.random.normal(shape=(1, Nkv, L, V))
            ref = mlx_primitives_sdpa(q, k, v, scale)
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
            self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_sdpa_vector_batched(self):
        D = 64
        q = mx.random.normal(shape=(2, 1, 3, D))
        k = mx.random.normal(shape=(2, 1, 3, D))
        v = mx.random.normal(shape=(2, 1, 3, D))

        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=None, scale=1.0)
        ref = mlx_ref_attn(q, k, v)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

        q = mx.random.normal(shape=(2, 4, 3, D))
        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=None, scale=1.0)
        ref = mlx_ref_attn(q, k, v)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

        q = mx.random.normal(shape=(2, 3, 4, D)).swapaxes(1, 2)
        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=None, scale=1.0)
        ref = mlx_ref_attn(q, k, v)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

        k = mx.random.normal(shape=(2, 3, 1, D)).swapaxes(1, 2)
        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=None, scale=1.0)
        ref = mlx_ref_attn(q, k, v)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

        q = mx.random.normal(shape=(2, 4, 3, D))
        k = mx.random.normal(shape=(2, 3, 2, D)).swapaxes(1, 2)
        v = mx.random.normal(shape=(2, 2, 3, D))
        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=None, scale=1.0)
        ref = mlx_ref_attn(q, k, v)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

        q = mx.random.normal(shape=(2, 4, 3, D))
        k = mx.random.normal(shape=(2, 1, 3, D))
        v = mx.random.normal(shape=(2, 1, 3, D))
        mask = 10 * mx.random.normal(shape=(1, 2, 3, 3)).swapaxes(0, 1)
        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=1.0)
        ref = mlx_ref_attn(q, k, v, mask=mask)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    @unittest.skipIf(not mx.is_available(mx.gpu), "too slow on CPU")
    def test_sdpa(self):
        # fmt: off
        shapes_64 = [
            # (  B,   qsl,   ksl, head_dim, n_qh, n_kvh)
            (  1,    20,    20,       64,    3,     3),
            (  1,    63,    63,       64,   24,    24),
            (  1,   129,   129,       64,   24,    24),
            (  1,   400,   400,       64,   24,    24),
            (  1,   128,   128,       64,   32,    32),
            (  1,    64,   128,       64,   32,    32),
            (  1,    65,   128,       64,   32,     8),
            (  1,    64,   127,       64,   32,     8),
            (  1,    65,   127,       64,   32,     8),
            (  1,   127,    65,       64,   32,     8),
        ]
        shapes_128 = [
            # (  B,   qsl,   ksl, head_dim, n_qh, n_kvh)
            (  1,   128,   128,      128,   32,     8),
            (  1,    64,   128,      128,   32,     8),
            (  1,    65,   127,      128,   32,     8),
            (  1,   127,    65,      128,   32,     8),
        ]
        for ksl in [7, 9, 32, 63, 67, 129, 400, 2000]:
            shapes_128.append((1, 1, ksl, 128, 32, 32))
            shapes_128.append((1, 1, ksl, 128, 32, 8))
        # fmt: on

        shapes = shapes_64 + shapes_128
        dtypes = [mx.float16]
        if mx.metal.is_available():
            dtypes.append(mx.float32)
        masks = [None, "additive", "bool", "causal"]
        transposes = (False, True)

        for dtype, t, mask_str, (B, qL, kL, D, qH, kH) in product(
            dtypes, transposes, masks, shapes
        ):
            with self.subTest(
                B=B,
                qsl=qL,
                ksl=kL,
                head_dim=D,
                n_q_heads=qH,
                n_kv_heads=kH,
                mask=mask_str,
                transpose=t,
                dtype=dtype,
            ):
                q, k, v, scale, mask = prepare_inputs(
                    B, qL, kL, D, qH, kH, mask_str, t, dtype
                )

                out_ref = do_attention(mlx_ref_attn, q, k, v, scale, mask, t)

                out_fst = do_attention(
                    mx.fast.scaled_dot_product_attention,
                    q,
                    k,
                    v,
                    scale,
                    mask,
                    t,
                )

                # For causal mask when qL > kL, first qL-kL rows are undefined
                # Compare only the valid portion
                if mask_str == "causal" and qL > kL:
                    offset = qL - kL
                    if t:  # transpose=True: shape is (B, qL, qH, D)
                        out_ref = out_ref[:, offset:, :, :]
                        out_fst = out_fst[:, offset:, :, :]
                    else:  # transpose=False: shape is (B, qH, qL, D)
                        out_ref = out_ref[:, :, offset:, :]
                        out_fst = out_fst[:, :, offset:, :]

                atol = 2e-5 if dtype == mx.float32 else 3e-4

                self.assertListEqual(list(out_ref.shape), list(out_fst.shape))

                diff = mx.abs(out_fst - out_ref) - atol * mx.abs(out_ref)
                self.assertLessEqual(mx.max(diff).item(), atol)

    def test_sdpa_broadcast_mask(self):
        mask = mx.array(True)
        D = 64
        Nq = 4
        Nkv = 1
        scale = 1.0
        L = 256

        mx.random.seed(0)
        q = 5e-1 * mx.random.normal(shape=(1, Nq, L, D))
        k = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        v = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        ref = mlx_primitives_sdpa(q, k, v, scale, mask=mask)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_sdpa_noncontiguous_inputs(self):
        mask = mx.ones(shape=(4, 1, 7, 7), dtype=mx.bool_)
        mx.random.seed(0)
        q = mx.random.normal(shape=(4, 7, 32, 64)).swapaxes(1, 2)

        k = mx.random.normal(shape=(4, 7, 8, 64)).swapaxes(1, 2)
        v = mx.random.normal(shape=(4, 7, 8, 64)).swapaxes(1, 2)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=mask)
        ref = mlx_ref_attn(q, k, v, scale=1.0, mask=mask)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_sdpa_promote_mask(self):
        mask = mx.array(2.0, mx.bfloat16)
        D = 64
        Nq = 4
        Nkv = 1
        scale = 1.0
        L = 256

        mx.random.seed(0)
        q = 5e-1 * mx.random.normal(shape=(1, Nq, L, D))
        k = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        v = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        ref = mlx_primitives_sdpa(q, k, v, scale, mask=mask)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_sdpa_nan_bug(self):
        N = 128
        q_shape = (1, 1, N, 128)
        kv_shape = (1, 1, N, 128)
        q = mx.random.uniform(shape=q_shape)
        k = mx.random.uniform(shape=kv_shape)
        v = mx.random.uniform(shape=kv_shape)

        # Make boolean window causal mask
        linds = rinds = mx.arange(N)
        linds = linds[:, None]
        rinds = rinds[None]
        mask = linds >= rinds
        mask = mask & (linds <= rinds + 111)

        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=1.0)
        expected = mlx_ref_attn(q, k, v, mask=mask, scale=1.0)
        self.assertFalse(mx.isnan(out).any().item())
        self.assertLessEqual(mx.abs(out - expected).max().item(), 1e-4)

        # And an additive one
        mask = mx.log(mask)

        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=1.0)
        expected = mlx_ref_attn(q, k, v, mask=mask, scale=1.0)
        self.assertFalse(mx.isnan(out).any().item())
        self.assertLessEqual(mx.abs(out - expected).max().item(), 1e-4)

    def test_sdpa_attention_sinks(self):
        B = 2
        N_q = N_kv = 8
        T_q = T_kv = 128
        D = 64

        q = mx.random.normal(shape=(B, N_q, T_q, D))
        k = mx.random.normal(shape=(B, N_kv, T_kv, D))
        v = mx.random.normal(shape=(B, N_kv, T_kv, D))
        scale = D**-0.5

        # sinks should promote to correct type
        sinks = mx.random.normal(shape=(N_q,))
        with self.assertRaises(ValueError):
            mx.fast.scaled_dot_product_attention(
                q.astype(mx.float16),
                k.astype(mx.float16),
                v.astype(mx.float16),
                scale=scale,
                sinks=sinks,
            )

        # Wrong shapes
        sinks = mx.random.normal(shape=(N_q + 1,))
        with self.assertRaises(ValueError):
            mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, sinks=sinks)

        sinks = mx.random.normal(shape=())
        with self.assertRaises(ValueError):
            mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, sinks=sinks)

        for T_q, T_kv, N_kv, dtype in product(
            (1, 128),
            (128, 4096),
            (2, 8),
            (mx.float16, mx.float32),
        ):
            with self.subTest(T_q=T_q, T_kv=T_kv, N_kv=N_kv, dtype=dtype):
                q = mx.random.normal(shape=(B, N_q, T_q, D), dtype=dtype)
                k = mx.random.normal(shape=(B, N_kv, T_kv, D), dtype=dtype)
                v = mx.random.normal(shape=(B, N_kv, T_kv, D), dtype=dtype)
                sinks = 10 * mx.random.normal(shape=(N_q,), dtype=dtype)

                expected = mlx_ref_attn(q, k, v, scale, sinks=sinks)
                out = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, sinks=sinks
                )
                atol = 1e-5 if dtype == mx.float32 else 1e-2
                self.assertTrue(mx.allclose(out, expected, atol=atol))

    def test_sdpa_grad(self):
        # High tolerance due to cuDNN SDPA kernel requiring tf32.
        tolerance = {"rtol": 1e-2, "atol": 1e-2}

        def test_vjp(slow, fast, primals):
            cotan = mx.ones_like(primals[0])
            o1, vjp1 = mx.vjp(slow, primals, [cotan])
            o2, vjp2 = mx.vjp(fast, primals, [cotan])

            self.assertTrue(mx.allclose(o1[0], o2[0], **tolerance))
            for i in range(3):
                self.assertTrue(mx.allclose(vjp1[i], vjp2[i], **tolerance))

        def test_grad(slow, fast, args):
            g1 = mx.grad(slow)(*args)
            g2 = mx.grad(fast)(*args)

            self.assertTrue(mx.allclose(g1, g2, **tolerance))

        B, N_kv, T, D = (2, 8, 128, 64)
        scale = D**-0.5

        for N_q in (8, 32):
            q = mx.random.normal(shape=(B, N_q, T, D), dtype=mx.float16)
            k = mx.random.normal(shape=(B, N_kv, T, D), dtype=mx.float16)
            v = mx.random.normal(shape=(B, N_kv, T, D), dtype=mx.float16)

            mask_additive = mx.random.normal((B, N_q, T, T), dtype=mx.float16)
            mask_bool = mx.random.uniform(0, 1, (B, N_q, T, T), dtype=mx.float16) < 0.5

            for mask in (None, "causal", mask_additive, mask_bool):
                sdpa_slow = lambda q, k, v: mlx_ref_attn(
                    q, k, v, scale=scale, mask=mask
                )
                sdpa_fast = lambda q, k, v: mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, mask=mask
                )
                test_vjp(sdpa_slow, sdpa_fast, [q, k, v])

                loss_slow = lambda q, k, v: mlx_ref_attn(
                    q, k, v, scale=scale, mask=mask
                ).sum()
                loss_fast = lambda q, k, v: mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, mask=mask
                ).sum()
                test_grad(loss_slow, loss_fast, [q, k, v])

    def test_sdpa_sliced(self):
        N = 8
        D = 64
        scale = D**-0.5

        for B, T_q, T_kv, offset, mask in product(
            (1, 2, 4),
            (1, 8),
            (256, 512),
            (8, 9, 64, 79),
            (None, "causal"),
        ):
            with self.subTest(B=B, T_q=T_q, T_kv=T_kv, offset=offset, mask=mask):
                q = mx.random.normal((B, N, T_q, D), mx.float16)
                k = mx.random.normal((B, N, T_kv, D), mx.float16)
                v = mx.random.normal((B, N, T_kv, D), mx.float16)

                k = k[..., :offset, :]
                v = v[..., :offset, :]

                ref = mlx_ref_attn(q, k, v, scale=scale, mask=mask)

                for i in range(2):
                    out = mx.fast.scaled_dot_product_attention(
                        q, k, v, scale=scale, mask=mask
                    )
                    if B == 1:
                        tolerance = {"rtol": 1e-3, "atol": 1e-3}
                    else:
                        tolerance = {"rtol": 1e-2, "atol": 1e-2}
                    self.assertTrue(mx.allclose(ref, out, **tolerance))


def varlen_ref_attn(
    q_packed, k_packed, v_packed, cu_seqlens_q, cu_seqlens_k, scale, do_causal=False
):
    """Reference implementation: split packed tensors by cu_seqlens, run standard SDPA on each."""
    num_seqs = len(cu_seqlens_q) - 1
    outputs = []
    n_q_heads = q_packed.shape[1]
    n_kv_heads = k_packed.shape[1]

    for i in range(num_seqs):
        q_s = cu_seqlens_q[i]
        q_e = cu_seqlens_q[i + 1]
        k_s = cu_seqlens_k[i]
        k_e = cu_seqlens_k[i + 1]

        # q_packed: (total_q, H, D) -> (seq_qL, H, D) -> (1, H, seq_qL, D)
        qi = q_packed[q_s:q_e]  # (seq_qL, H, D)
        qi = mx.expand_dims(qi, 0).transpose(0, 2, 1, 3)  # (1, H, seq_qL, D)
        ki = k_packed[k_s:k_e]
        ki = mx.expand_dims(ki, 0).transpose(0, 2, 1, 3)
        vi = v_packed[k_s:k_e]
        vi = mx.expand_dims(vi, 0).transpose(0, 2, 1, 3)

        mask = "causal" if do_causal else None
        oi = mlx_ref_attn(qi, ki, vi, scale=scale, mask=mask)
        # (1, H, seq_qL, D) -> (seq_qL, H, D)
        oi = oi.transpose(0, 2, 1, 3).squeeze(0)
        outputs.append(oi)

    return mx.concatenate(outputs, axis=0)


class TestFastSDPAVarlen(mlx_tests.MLXTestCase):

    def test_varlen_matches_individual(self):
        """Pack 3 seqs, compare to individual SDPA."""
        mx.random.seed(42)
        seq_lens = [17, 32, 43]
        dtypes = [mx.float16, mx.float32]
        head_dims = [64, 80, 128]

        for dtype in dtypes:
            for D in head_dims:
                with self.subTest(dtype=dtype, D=D):
                    H = 8
                    total_q = sum(seq_lens)
                    cu_seqlens = mx.array(
                        [0] + [int(x) for x in np.cumsum(seq_lens)], dtype=mx.int32
                    )

                    q = mx.random.normal(shape=(total_q, H, D)).astype(dtype)
                    k = mx.random.normal(shape=(total_q, H, D)).astype(dtype)
                    v = mx.random.normal(shape=(total_q, H, D)).astype(dtype)
                    scale = D**-0.5

                    out = mx.fast.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        scale=scale,
                        cu_seqlens_q=cu_seqlens,
                        cu_seqlens_k=cu_seqlens,
                    )
                    ref = varlen_ref_attn(
                        q, k, v, cu_seqlens.tolist(), cu_seqlens.tolist(), scale
                    )

                    atol = 1e-2 if dtype == mx.float16 else 1e-4
                    self.assertTrue(
                        mx.allclose(out, ref, atol=atol, rtol=atol),
                        f"Mismatch for dtype={dtype}, D={D}, max_diff={mx.max(mx.abs(out - ref)).item()}",
                    )

    def test_varlen_causal(self):
        """Per-sequence causal masking."""
        mx.random.seed(42)
        seq_lens = [17, 32, 43]
        D = 64
        H = 8
        total_q = sum(seq_lens)
        cu_seqlens = mx.array(
            [0] + [int(x) for x in np.cumsum(seq_lens)], dtype=mx.int32
        )

        for dtype in [mx.float16, mx.float32]:
            with self.subTest(dtype=dtype):
                q = mx.random.normal(shape=(total_q, H, D)).astype(dtype)
                k = mx.random.normal(shape=(total_q, H, D)).astype(dtype)
                v = mx.random.normal(shape=(total_q, H, D)).astype(dtype)
                scale = D**-0.5

                out = mx.fast.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    scale=scale,
                    mask="causal",
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                )
                ref = varlen_ref_attn(
                    q,
                    k,
                    v,
                    cu_seqlens.tolist(),
                    cu_seqlens.tolist(),
                    scale,
                    do_causal=True,
                )

                atol = 1e-2 if dtype == mx.float16 else 1e-4
                self.assertTrue(
                    mx.allclose(out, ref, atol=atol, rtol=atol),
                    f"Mismatch for dtype={dtype}, max_diff={mx.max(mx.abs(out - ref)).item()}",
                )

    def test_varlen_no_causal(self):
        """Bidirectional within each sequence."""
        mx.random.seed(42)
        seq_lens = [20, 30]
        D = 64
        H = 4
        total_q = sum(seq_lens)
        cu_seqlens = mx.array(
            [0] + [int(x) for x in np.cumsum(seq_lens)], dtype=mx.int32
        )

        q = mx.random.normal(shape=(total_q, H, D)).astype(mx.float32)
        k = mx.random.normal(shape=(total_q, H, D)).astype(mx.float32)
        v = mx.random.normal(shape=(total_q, H, D)).astype(mx.float32)
        scale = D**-0.5

        out = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=scale,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
        )
        ref = varlen_ref_attn(q, k, v, cu_seqlens.tolist(), cu_seqlens.tolist(), scale)

        self.assertTrue(mx.allclose(out, ref, atol=1e-4, rtol=1e-4))

    def test_varlen_gqa(self):
        """GQA: n_q_heads=16, n_kv_heads=4."""
        mx.random.seed(42)
        seq_lens = [20, 30]
        D = 64
        n_q_heads = 16
        n_kv_heads = 4
        total_q = sum(seq_lens)
        cu_seqlens = mx.array(
            [0] + [int(x) for x in np.cumsum(seq_lens)], dtype=mx.int32
        )

        q = mx.random.normal(shape=(total_q, n_q_heads, D)).astype(mx.float32)
        k = mx.random.normal(shape=(total_q, n_kv_heads, D)).astype(mx.float32)
        v = mx.random.normal(shape=(total_q, n_kv_heads, D)).astype(mx.float32)
        scale = D**-0.5

        out = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=scale,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
        )
        ref = varlen_ref_attn(q, k, v, cu_seqlens.tolist(), cu_seqlens.tolist(), scale)

        self.assertTrue(mx.allclose(out, ref, atol=1e-4, rtol=1e-4))

    def test_varlen_single_sequence(self):
        """cu_seqlens=[0, L] matches standard 4D SDPA."""
        mx.random.seed(42)
        L = 64
        D = 64
        H = 8
        cu_seqlens = mx.array([0, L], dtype=mx.int32)

        q = mx.random.normal(shape=(L, H, D)).astype(mx.float32)
        k = mx.random.normal(shape=(L, H, D)).astype(mx.float32)
        v = mx.random.normal(shape=(L, H, D)).astype(mx.float32)
        scale = D**-0.5

        out_varlen = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=scale,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
        )

        # Standard 4D path
        q4d = q.reshape(1, L, H, D).transpose(0, 2, 1, 3)
        k4d = k.reshape(1, L, H, D).transpose(0, 2, 1, 3)
        v4d = v.reshape(1, L, H, D).transpose(0, 2, 1, 3)
        out_4d = mx.fast.scaled_dot_product_attention(
            q4d,
            k4d,
            v4d,
            scale=scale,
        )
        out_4d = out_4d.transpose(0, 2, 1, 3).squeeze(0)

        self.assertTrue(mx.allclose(out_varlen, out_4d, atol=1e-4, rtol=1e-4))

    def test_varlen_single_token_sequences(self):
        """Length-1 sequences in a pack."""
        mx.random.seed(42)
        num_seqs = 5
        D = 64
        H = 4
        total_q = num_seqs
        cu_seqlens = mx.array(list(range(num_seqs + 1)), dtype=mx.int32)

        q = mx.random.normal(shape=(total_q, H, D)).astype(mx.float32)
        k = mx.random.normal(shape=(total_q, H, D)).astype(mx.float32)
        v = mx.random.normal(shape=(total_q, H, D)).astype(mx.float32)
        scale = D**-0.5

        out = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=scale,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
        )
        # For single-token sequences, output should just be v (softmax of single element = 1)
        ref = varlen_ref_attn(q, k, v, cu_seqlens.tolist(), cu_seqlens.tolist(), scale)
        self.assertTrue(mx.allclose(out, ref, atol=1e-4, rtol=1e-4))

    def test_varlen_unaligned_lengths(self):
        """Lengths NOT multiples of BQ=32/BK (primes: 17, 31, 43, 67)."""
        mx.random.seed(42)
        seq_lens = [17, 31, 43, 67]
        D = 128
        H = 8
        total_q = sum(seq_lens)
        cu_seqlens = mx.array(
            [0] + [int(x) for x in np.cumsum(seq_lens)], dtype=mx.int32
        )

        for dtype in [mx.float16, mx.float32]:
            with self.subTest(dtype=dtype):
                q = mx.random.normal(shape=(total_q, H, D)).astype(dtype)
                k = mx.random.normal(shape=(total_q, H, D)).astype(dtype)
                v = mx.random.normal(shape=(total_q, H, D)).astype(dtype)
                scale = D**-0.5

                out = mx.fast.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    scale=scale,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                )
                ref = varlen_ref_attn(
                    q, k, v, cu_seqlens.tolist(), cu_seqlens.tolist(), scale
                )

                atol = 1e-2 if dtype == mx.float16 else 1e-4
                self.assertTrue(
                    mx.allclose(out, ref, atol=atol, rtol=atol),
                    f"Mismatch for dtype={dtype}, max_diff={mx.max(mx.abs(out - ref)).item()}",
                )

    def test_varlen_invalid_inputs(self):
        """Error on wrong dtype, non-matching counts, 4D input."""
        D = 64
        H = 4
        q = mx.random.normal(shape=(10, H, D))
        k = mx.random.normal(shape=(10, H, D))
        v = mx.random.normal(shape=(10, H, D))
        scale = 1.0

        # Wrong dtype for cu_seqlens
        with self.assertRaises(ValueError):
            mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                cu_seqlens_q=mx.array([0, 10], dtype=mx.float32),
                cu_seqlens_k=mx.array([0, 10], dtype=mx.int32),
            )

        # Mismatched number of sequences
        with self.assertRaises(ValueError):
            mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                cu_seqlens_q=mx.array([0, 5, 10], dtype=mx.int32),
                cu_seqlens_k=mx.array([0, 10], dtype=mx.int32),
            )

        # 4D input (should require 3D for varlen)
        q4d = mx.random.normal(shape=(1, H, 10, D))
        k4d = mx.random.normal(shape=(1, H, 10, D))
        v4d = mx.random.normal(shape=(1, H, 10, D))
        with self.assertRaises(ValueError):
            mx.fast.scaled_dot_product_attention(
                q4d,
                k4d,
                v4d,
                scale=scale,
                cu_seqlens_q=mx.array([0, 10], dtype=mx.int32),
                cu_seqlens_k=mx.array([0, 10], dtype=mx.int32),
            )

        # cu_seqlens_q without cu_seqlens_k
        with self.assertRaises(ValueError):
            mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                cu_seqlens_q=mx.array([0, 10], dtype=mx.int32),
            )

        # Array mask not supported with cu_seqlens
        with self.assertRaises(ValueError):
            mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                mask=mx.ones((10, 10)),
                cu_seqlens_q=mx.array([0, 10], dtype=mx.int32),
                cu_seqlens_k=mx.array([0, 10], dtype=mx.int32),
            )

    def test_varlen_gradient(self):
        """VJP through packed matches VJP through individual."""
        mx.random.seed(42)
        seq_lens = [16, 32]
        D = 64
        H = 4
        total_q = sum(seq_lens)
        cu_seqlens = mx.array(
            [0] + [int(x) for x in np.cumsum(seq_lens)], dtype=mx.int32
        )
        scale = D**-0.5

        q = mx.random.normal(shape=(total_q, H, D)).astype(mx.float32)
        k = mx.random.normal(shape=(total_q, H, D)).astype(mx.float32)
        v = mx.random.normal(shape=(total_q, H, D)).astype(mx.float32)

        def loss_varlen(q, k, v):
            out = mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
            )
            return out.sum()

        def loss_ref(q, k, v):
            out = varlen_ref_attn(
                q, k, v, cu_seqlens.tolist(), cu_seqlens.tolist(), scale
            )
            return out.sum()

        g_varlen = mx.grad(loss_varlen)(q, k, v)
        g_ref = mx.grad(loss_ref)(q, k, v)

        self.assertTrue(
            mx.allclose(g_varlen, g_ref, atol=1e-2, rtol=1e-2),
            f"Gradient mismatch, max_diff={mx.max(mx.abs(g_varlen - g_ref)).item()}",
        )

    def test_varlen_stress_many_sequences(self):
        """Stress test with 100+ length-1 sequences to exercise seq_block_map."""
        mx.random.seed(42)
        num_seqs = 128
        D = 64
        H = 4
        total_q = num_seqs
        cu_seqlens = mx.array(list(range(num_seqs + 1)), dtype=mx.int32)

        q = mx.random.normal(shape=(total_q, H, D)).astype(mx.float32)
        k = mx.random.normal(shape=(total_q, H, D)).astype(mx.float32)
        v = mx.random.normal(shape=(total_q, H, D)).astype(mx.float32)
        scale = D**-0.5

        out = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=scale,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
        )
        ref = varlen_ref_attn(q, k, v, cu_seqlens.tolist(), cu_seqlens.tolist(), scale)
        self.assertTrue(mx.allclose(out, ref, atol=1e-4, rtol=1e-4))

        # Also test with mixed lengths (many short + a few long)
        seq_lens = [1] * 50 + [32] * 10 + [64] * 5
        total_q = sum(seq_lens)
        cu_seqlens = mx.array(
            [0] + [int(x) for x in np.cumsum(seq_lens)], dtype=mx.int32
        )

        q = mx.random.normal(shape=(total_q, H, D)).astype(mx.float32)
        k = mx.random.normal(shape=(total_q, H, D)).astype(mx.float32)
        v = mx.random.normal(shape=(total_q, H, D)).astype(mx.float32)

        out = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=scale,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
        )
        ref = varlen_ref_attn(q, k, v, cu_seqlens.tolist(), cu_seqlens.tolist(), scale)
        self.assertTrue(mx.allclose(out, ref, atol=1e-4, rtol=1e-4))

    def test_varlen_bfloat16(self):
        """Test bfloat16 dtype — kernel is instantiated, verify correctness."""
        mx.random.seed(42)
        seq_lens = [20, 32, 48]
        D = 64
        H = 8
        total_q = sum(seq_lens)
        cu_seqlens = mx.array(
            [0] + [int(x) for x in np.cumsum(seq_lens)], dtype=mx.int32
        )

        q = mx.random.normal(shape=(total_q, H, D)).astype(mx.bfloat16)
        k = mx.random.normal(shape=(total_q, H, D)).astype(mx.bfloat16)
        v = mx.random.normal(shape=(total_q, H, D)).astype(mx.bfloat16)
        scale = D**-0.5

        for mask in [None, "causal"]:
            with self.subTest(mask=mask):
                out = mx.fast.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    scale=scale,
                    mask=mask,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                )
                ref = varlen_ref_attn(
                    q,
                    k,
                    v,
                    cu_seqlens.tolist(),
                    cu_seqlens.tolist(),
                    scale,
                    do_causal=(mask == "causal"),
                )
                self.assertTrue(
                    mx.allclose(out, ref, atol=1e-2, rtol=1e-2),
                    f"Mismatch for bf16 mask={mask}, max_diff={mx.max(mx.abs(out - ref)).item()}",
                )

    def test_varlen_large_total_tokens(self):
        """Large total_tokens (16K+) for real-world LLM batch sizes."""
        mx.random.seed(42)
        # Simulate a realistic batch: 16 sequences of ~1024 tokens each
        seq_lens = [
            1024,
            512,
            768,
            1280,
            896,
            1024,
            640,
            1024,
            512,
            1024,
            768,
            640,
            1024,
            896,
            512,
            1024,
        ]
        D = 128
        H = 32
        n_kv = 8
        total_q = sum(seq_lens)  # ~13,548
        cu_seqlens = mx.array(
            [0] + [int(x) for x in np.cumsum(seq_lens)], dtype=mx.int32
        )

        q = mx.random.normal(shape=(total_q, H, D)).astype(mx.float16)
        k = mx.random.normal(shape=(total_q, n_kv, D)).astype(mx.float16)
        v = mx.random.normal(shape=(total_q, n_kv, D)).astype(mx.float16)
        scale = D**-0.5

        out = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=scale,
            mask="causal",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
        )
        # Just verify it doesn't crash and output has correct shape
        self.assertEqual(out.shape, (total_q, H, D))
        self.assertFalse(mx.any(mx.isnan(out)).item())

        # Spot-check first sequence against reference
        ref = varlen_ref_attn(
            q, k, v, cu_seqlens.tolist(), cu_seqlens.tolist(), scale, do_causal=True
        )
        first_len = seq_lens[0]
        self.assertTrue(
            mx.allclose(out[:first_len], ref[:first_len], atol=1e-2, rtol=1e-2),
            f"First seq mismatch, max_diff={mx.max(mx.abs(out[:first_len] - ref[:first_len])).item()}",
        )


if __name__ == "__main__":
    mlx_tests.MLXTestRunner(failfast=True)
