// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0
//
// Minimal CHOLMOD wrapper mirroring the C++ `math/suitesparse.h`: int32 CSC
// indices, lower-triangle symmetric input, `cholmod_analyze` + `factorize` +
// `solve`. Float and double scalars are supported.

#include <cholmod.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  cholmod_common common;
  cholmod_factor *factor;
  int *p;
  int *i;
  size_t nnz;
  int n;
  int single;
} tinyopt_cholmod_ctx;

void *tinyopt_cholmod_create(int single) {
  tinyopt_cholmod_ctx *ctx = (tinyopt_cholmod_ctx *)calloc(1, sizeof(*ctx));
  if (!ctx) return NULL;
  if (!cholmod_start(&ctx->common)) {
    free(ctx);
    return NULL;
  }
  ctx->common.print = 0;
  ctx->single = single;
  return ctx;
}

void tinyopt_cholmod_destroy(void *ptr) {
  tinyopt_cholmod_ctx *ctx = (tinyopt_cholmod_ctx *)ptr;
  if (!ctx) return;
  if (ctx->factor) cholmod_free_factor(&ctx->factor, &ctx->common);
  free(ctx->p);
  free(ctx->i);
  cholmod_finish(&ctx->common);
  free(ctx);
}

static int copy_indices(tinyopt_cholmod_ctx *ctx, int n, const int64_t *col_ptr,
                        const int64_t *row_idx, size_t nnz) {
  free(ctx->p);
  free(ctx->i);
  ctx->p = (int *)malloc(sizeof(int) * (size_t)(n + 1));
  ctx->i = (int *)malloc(sizeof(int) * (nnz > 0 ? nnz : 1));
  if (!ctx->p || !ctx->i) return 0;
  for (int j = 0; j <= n; ++j) {
    if (col_ptr[j] < 0 || col_ptr[j] > (int64_t)nnz) return 0;
    ctx->p[j] = (int)col_ptr[j];
  }
  for (size_t k = 0; k < nnz; ++k) {
    if (row_idx[k] < 0 || row_idx[k] >= (int64_t)n) return 0;
    ctx->i[k] = (int)row_idx[k];
  }
  ctx->n = n;
  ctx->nnz = nnz;
  return 1;
}

static void fill_sparse(tinyopt_cholmod_ctx *ctx, cholmod_sparse *A, const void *values,
                        int pattern) {
  memset(A, 0, sizeof(*A));
  A->nrow = (size_t)ctx->n;
  A->ncol = (size_t)ctx->n;
  A->nzmax = ctx->nnz;
  A->p = ctx->p;
  A->i = ctx->i;
  A->x = (void *)values;
  A->z = NULL;
  A->nz = NULL;
  A->stype = -1;  // lower triangle of a symmetric matrix
  A->itype = CHOLMOD_INT;
  A->xtype = pattern ? CHOLMOD_PATTERN : CHOLMOD_REAL;
  A->dtype = ctx->single ? CHOLMOD_SINGLE : CHOLMOD_DOUBLE;
  A->sorted = 1;
  A->packed = 1;
}

int tinyopt_cholmod_analyze(void *ptr, int n, const int64_t *col_ptr, const int64_t *row_idx,
                            size_t nnz) {
  tinyopt_cholmod_ctx *ctx = (tinyopt_cholmod_ctx *)ptr;
  if (!ctx || n <= 0) return 0;
  if (!copy_indices(ctx, n, col_ptr, row_idx, nnz)) return 0;
  cholmod_sparse A;
  fill_sparse(ctx, &A, NULL, 1);
  if (ctx->factor) cholmod_free_factor(&ctx->factor, &ctx->common);
  ctx->factor = cholmod_analyze(&A, &ctx->common);
  return ctx->factor != NULL;
}

int tinyopt_cholmod_factorize(void *ptr, const void *values) {
  tinyopt_cholmod_ctx *ctx = (tinyopt_cholmod_ctx *)ptr;
  if (!ctx || !ctx->factor) return 0;
  cholmod_sparse A;
  fill_sparse(ctx, &A, values, 0);
  if (!cholmod_factorize(&A, ctx->factor, &ctx->common)) return 0;
  return ctx->factor->minor == (size_t)ctx->n;
}

int tinyopt_cholmod_solve(void *ptr, const void *rhs, void *out) {
  tinyopt_cholmod_ctx *ctx = (tinyopt_cholmod_ctx *)ptr;
  if (!ctx || !ctx->factor) return 0;
  cholmod_dense b;
  memset(&b, 0, sizeof(b));
  b.nrow = (size_t)ctx->n;
  b.ncol = 1;
  b.nzmax = (size_t)ctx->n;
  b.d = (size_t)ctx->n;
  b.x = (void *)rhs;
  b.z = NULL;
  b.xtype = CHOLMOD_REAL;
  b.dtype = ctx->single ? CHOLMOD_SINGLE : CHOLMOD_DOUBLE;
  cholmod_dense *x = cholmod_solve(CHOLMOD_A, ctx->factor, &b, &ctx->common);
  if (!x) return 0;
  size_t bytes = (size_t)ctx->n * (ctx->single ? sizeof(float) : sizeof(double));
  memcpy(out, x->x, bytes);
  cholmod_free_dense(&x, &ctx->common);
  return 1;
}
