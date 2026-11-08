/* ---- GC root liveness on a graph of the emitted C (the CFA prototype) ----

   A root exists so a collection can find a value only the C stack holds.
   The collection needs it only if the value is read after the collection.
   A pointer whose reads all come before every point where a collection can
   start needs no root. That is liveness, a backward dataflow problem on the
   control-flow graph: a variable is live at a point if some path from the
   point reads it before it is written. A root is needed if and only if its
   variable is live just after some collection point.

       def insert(h, key)                 sp_RBNode *_t2 = h->iv_left;
         ...                              SP_GC_ROOT(_t2);   <- dead
         h.left = insert(h.left, key)     h->iv_left = sp_insert(_t2, key);

   The callee gets _t2 as its parameter and roots that itself when it needs
   to, and the caller reads _t2 no more. h is read after the call (the store
   goes into it), so its root stays.

   Only codegen's own temporaries lose roots that way. A variable Ruby code
   names (a local or parameter, lv_; a captured one, _cell_; self) keeps
   its root wherever the function can collect, as CRuby keeps the object a
   local names alive until the method returns: `keep = R.new` with a
   finalizer must not run it early, and a long-lived tree no later line
   reads must not leave the heap (gcbench went from 96 collections to
   3,996 when it did).

   The graph is built from the function's emitted C, not from the Ruby tree:
   most roots belong to temporaries the emitters make (_t2 above), which the
   Ruby tree never names, and the points where a collection can start are C
   calls. The pass reads the function the way M1', M2 and the frame builder
   do: after the body is emitted, before the frame is built.

   What the graph holds, per block, in the order the C runs it:
   - USE v: v's value is read. A read inside an expression counts at the
     end of the expression's siblings when C does not fix the order
     (`f(a, g())` may read a after g runs), so a value a sibling holds is
     live across every call in the other siblings.
   - DEF v: v is written as a whole (`v = e;`, `T v = e;`), after e.
   - COL: a call that can start a collection and then return. A call is
     not one when M2's list (gc_leaf_call_ok) has it, when it never returns
     (a raise: what it collects is never read here again), or when it is a
     method this pass already read and found has no COL.
   After a COL, the call's own arguments still count as read, unless the
   callee keeps them alive itself: a method this pass read that roots the
   parameter first thing and changes it only by whole writes, with no copy
   of the old value left live (gl_safe_args). A runtime function's
   arguments count as read (sp_X_new allocates before it roots its
   arguments), and so does a method's self, which no method roots.

   A copy keeps its source alive: `__typeof__(h) _wb1 = h;` puts h in an
   unrooted temporary that the store after the call reads, so a read of an
   unrooted pointer counts as a read of every variable its writes read.

   Anything the pass cannot read keeps every root: setjmp, a jump it does
   not know, the root depth touched directly, an unrooted pointer whose
   address is taken. A rooted variable whose address is taken keeps its
   root. SP_NO_ROOT_LIVE=1 turns the pass off; SP_ROOT_LIVE_LOG=1 logs
   each function's roots. */
#include "codegen_internal.h"
#include "decide.h"
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { GT_ID = 1, GT_NUM, GT_STR, GT_CHR, GT_P };
typedef struct { int k, off, len; } GTok;

enum { GE_USE = 1, GE_DEF, GE_COL };
typedef struct { int k, v; } GEv;

typedef struct {
  GEv *ev; int nev, cev;
  int *succ; int nsucc, csucc;
} GBlk;

typedef struct {
  int off, len;           /* the name, in the text */
  char ptr;               /* its type can hold a GC reference */
  char rooted;            /* a root statement names it: 1 right after the
                             declaration (nothing collects between), 2 later */
  char addr;              /* its address is taken */
  char par;               /* a parameter of the function */
  char keep;              /* static storage: its root is not a stack root */
  char need;              /* live just after a collection point */
  char wr;                /* written after its declaration: 1 by whole
                             statements `v = e;`, 2 any other way */
  int *src; int nsrc, csrc;   /* the variables its writes read */
  int dblk, dcol;         /* the block and the collection count at its declaration */
} GVar;

typedef struct { int *v; int n, c; int col; } GInfo;
typedef struct { int beg, end, var; } GRoot;     /* token range of a root statement */
typedef struct { int off, len, blk; } GLabel;

typedef struct {
  const char *m;          /* the function's text, literals and comments blanked */
  GTok *t; int nt, ct;
  int *mt;                /* the matching bracket of each bracket token */
  int i;                  /* statement cursor */
  GBlk *b; int nb, cb;
  int cur, exitb;
  GVar *v; int nv, cv;
  int *sv; int nsv, csv;  /* the visible variables, innermost last */
  int sm[1024]; int nsm;
  int brk[256], nbrk, cnt[256], ncnt;
  int swh[128], swdef[128], nsw;
  GLabel *lab; int nlab, clab;
  GRoot *r; int nr, cr;
  GInfo *ci;              /* the collector the events report to */
  const char *bail;
  int ncol;
  int params_lo, params_hi;   /* the signature's parameter tokens */
  int entryb;             /* the block the body starts in */
  int np; int *pvar;      /* the parameters' variables (-1: none) */
  char *cons;             /* a caller's argument is safe in parameter k */
  int self_ce;            /* this function's entry in the callee table */
  int selfcall;           /* the body calls the function itself */
  char *ent;              /* parameter k is rooted at the entry, first thing */
  char *leak;             /* a write of parameter k leaves a copy of its old value live */
  int cond, cond_mark;    /* inside a part of an expression that may not run */
} GFn;

/* ---- what the pass knows about the functions it has read ---- */

typedef struct { char *name; char leaf; int np; char *cons; } GCallee;
/* leaf: a call to it starts no collection. cons[k]: an argument for
   parameter k needs nothing from the caller after the call starts, because
   it holds no reference, or the callee roots the parameter before anything
   can collect and keeps the value in it (gl_safe_args). np 0 knows no
   parameter. */
static GCallee *g_cal; static int g_ncal, g_ccal;
static int *g_calh; static int g_calhn;

static unsigned gl_hash(const char *s, size_t n) {
  unsigned h = 2166136261u;
  for (size_t i = 0; i < n; i++) h = (h ^ (unsigned char)s[i]) * 16777619u;
  return h;
}
static GCallee *gl_callee(const char *s, size_t n) {
  if (!g_calhn) return NULL;
  for (unsigned h = gl_hash(s, n) & (unsigned)(g_calhn - 1); g_calh[h] >= 0; h = (h + 1) & (unsigned)(g_calhn - 1)) {
    GCallee *e = &g_cal[g_calh[h]];
    if (strlen(e->name) == n && !memcmp(e->name, s, n)) return e;
  }
  return NULL;
}
/* A new program: no function read yet. */
void gc_live_reset(void) {
  for (int i = 0; i < g_ncal; i++) { free(g_cal[i].name); free(g_cal[i].cons); }
  free(g_cal); free(g_calh);
  g_cal = NULL; g_calh = NULL; g_ncal = g_ccal = g_calhn = 0;
}
/* The entry of a function the pass is about to read. It knows nothing yet:
   a call to the function from its own body (a recursion) collects, and its
   arguments count as read after it, except the ones the body's first
   statements root (gl_root_first). A second body under a name the pass has
   read makes the name unknown for good, and answers -1. */
static int gl_callee_open(const char *name, int np) {
  size_t n = strlen(name);
  GCallee *old = gl_callee(name, n);
  if (old) { old->leaf = 0; old->np = 0; return -1; }
  if (g_ncal == g_ccal) {
    g_ccal = g_ccal ? g_ccal * 2 : 64;
    g_cal = realloc(g_cal, sizeof *g_cal * (size_t)g_ccal);
    if (!g_cal) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  }
  GCallee *e = &g_cal[g_ncal];
  e->name = malloc(n + 1);
  e->leaf = 0; e->np = np;
  e->cons = calloc((size_t)(np > 0 ? np : 1), 1);
  if (!e->name || !e->cons) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  memcpy(e->name, name, n + 1);
  g_ncal++;
  if (g_ncal * 2 > g_calhn) {
    int hn = g_calhn ? g_calhn * 2 : 128;
    while (g_ncal * 2 > hn) hn *= 2;
    free(g_calh);
    g_calh = malloc(sizeof *g_calh * (size_t)hn);
    if (!g_calh) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
    g_calhn = hn;
    for (int i = 0; i < hn; i++) g_calh[i] = -1;
    for (int k = 0; k < g_ncal; k++) {
      unsigned h = gl_hash(g_cal[k].name, strlen(g_cal[k].name)) & (unsigned)(hn - 1);
      while (g_calh[h] >= 0) h = (h + 1) & (unsigned)(hn - 1);
      g_calh[h] = k;
    }
  }
  else {
    unsigned h = gl_hash(name, n) & (unsigned)(g_calhn - 1);
    while (g_calh[h] >= 0) h = (h + 1) & (unsigned)(g_calhn - 1);
    g_calh[h] = g_ncal - 1;
  }
  return g_ncal - 1;
}

static int gl_in(const char *const *list, const char *id, size_t n) {
  for (int i = 0; list[i]; i++) if (strlen(list[i]) == n && !strncmp(id, list[i], n)) return 1;
  return 0;
}
/* The runtime's raises: declared SP_NORETURN (lib/), they leave the
   function, so nothing here reads a value after what they collect. */
static int gl_noreturn(const char *id, size_t n) {
  static const char *const L[] = {
    "sp_abort_raise", "sp_bm_arg_mismatch", "sp_brk_throw", "sp_complex_reject_bool",
    "sp_dir_raise_closed", "sp_exc_abort_trap", "sp_exit_raise", "sp_fiber_raise_kill_self",
    "sp_file_raise_errno", "sp_float_arg_range_error", "sp_io_copy_stream_raise",
    "sp_io_raise_closed", "sp_nil_recv", "sp_nil_recv_raise", "sp_pm_raise_nomatch",
    "sp_poly_nil_nomethod", "sp_poly_recur_raise", "sp_poly_typed_hash_store_miss",
    "sp_pread_raise_errno", "sp_process_spawn_fail", "sp_raise_arity", "sp_raise_cannot_freeze",
    "sp_rbs_ary_fail",   /* not marked, but its body is one sp_raise_cls */
    "sp_raise_cls", "sp_raise_float_cmp", "sp_raise_frozen_array_rv", "sp_raise_frozen_str",
    "sp_raise_hash_nil_value", "sp_raise_hash_writeback_kind", "sp_raise_key_not_found",
    "sp_raise_kw_error", "sp_raise_load_error", "sp_raise_nil_cmp", "sp_raise_nil_float_op",
    "sp_raise_nil_opnd", "sp_raise_nil_to_float", "sp_raise_nil_to_int",
    "sp_raise_no_str_conversion", "sp_raise_nomethod_call", "sp_raise_nomethod_call_rest",
    "sp_raise_nomethod_obj", "sp_raise_poly", "sp_raise_poly_msg", "sp_raise_poly_msg_h",
    "sp_raise_poly_nomethod", "sp_raise_resolution_error", "sp_raise_stop_iteration",
    "sp_raise_typed_elem", "sp_raise_typed_hash_part", "sp_raise_writeback_kind",
    "sp_scan_bad_pattern", "sp_sig_default_raise", "sp_slot_store_mismatch",
    "sp_sock_raise_wait", "sp_sort_cmp_failed", "sp_stack_too_deep",
    "sp_typed_replace_elem_error", NULL };
  return gl_in(L, id, n);
}
/* Calls on top of M2's list that start no collection: an atomic load (the
   poll's test), the typed pointer Array's read (a bounds compare and a
   load; its _m form raises on nil), and the --rbs Array check (a tag test,
   and on a mismatch a raise). */
static int gl_pure_extra(const char *id, size_t n) {
  static const char *const L[] = { "SP_ATOMIC_LOAD", "sp_PtrArray_get", "sp_PtrArray_get_m",
                                   "sp_rbs_need_ary", "SP_RBS_CHECK_ARY", NULL };
  return gl_in(L, id, n);
}
/* Runtime functions that can collect, but root their arguments before they
   do: the poly to typed Array reads give the payload back, or convert in a
   function that roots the source first (sp_poly_as_int_array_conv). */
static int gl_roots_args(const char *id, size_t n) {
  static const char *const L[] = { "sp_poly_as_int_array", "sp_poly_as_float_array", NULL };
  return gl_in(L, id, n);
}
/* A macro (a name in capitals) can assign a variable it is given. These
   only read theirs. */
static int gl_macro(const char *id, size_t n) {
  static const char *const RO[] = { "SP_LIKELY", "SP_UNLIKELY", "SP_ATOMIC_LOAD", "SP_STR_B", "SP_WBO",
                                    "SP_RBS_CHECK_ARY", NULL };
  int up = 0;
  for (size_t i = 0; i < n; i++) {
    if (islower((unsigned char)id[i])) return 0;
    if (isupper((unsigned char)id[i])) up = 1;
  }
  return up && !gl_in(RO, id, n);
}
/* `(` after these opens no call: the operand is not evaluated */
static int gl_uneval(const char *id, size_t n) {
  static const char *const L[] = { "sizeof", "__typeof__", "__typeof", "typeof", "_Alignof",
                                   "__alignof__", "__builtin_offsetof", "offsetof",
                                   "__builtin_types_compatible_p", "__builtin_constant_p", NULL };
  return gl_in(L, id, n);
}
/* types that hold no GC reference */
static int gl_scalar_type(const char *id, size_t n) {
  static const char *const L[] = {
    "sp_int", "sp_bool", "sp_oint", "sp_ofloat", "sp_float", "double", "float", "int", "long",
    "short", "char", "unsigned", "signed", "_Bool", "bool", "size_t", "ssize_t", "int8_t",
    "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "intptr_t", "uintptr_t", "ptrdiff_t", "const", "volatile", "register", "static",
    "SP_UNUSED", NULL };
  return gl_in(L, id, n);
}
/* identifiers that end the reading: the function is not one this pass can
   follow, so every root stays */
static int gl_bail_id(const char *id, size_t n) {
  static const char *const L[] = { "setjmp", "_setjmp", "sigsetjmp", "__builtin_setjmp",
                                   "longjmp", "_longjmp", "siglongjmp", "sp_gc_nroots", "asm",
                                   "__asm__", "__asm", "__label__", "SP_CLEANUP", "cleanup",
                                   "SP_GC_ROOT_FRAME", "_gcf", NULL };
  if (gl_in(L, id, n)) return 1;
  return n > 14 && !strncmp(id, "_sp_main_part_", 14);
}

/* ---- tokens ---- */

static int gl_idch(char ch) { return isalnum((unsigned char)ch) || ch == '_' || ch == '$'; }

static void gl_tok(GFn *F, int k, size_t off, size_t len) {
  if (F->nt == F->ct) {
    F->ct = F->ct ? F->ct * 2 : 1024;
    F->t = realloc(F->t, sizeof *F->t * (size_t)F->ct);
    if (!F->t) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  }
  F->t[F->nt].k = k; F->t[F->nt].off = (int)off; F->t[F->nt].len = (int)len;
  F->nt++;
}
/* Tokens from the signature to the brace that closes the body. */
static int gl_tokenize(GFn *F, const char *m, size_t n) {
  static const char *const P3[] = { "<<=", ">>=", "...", NULL };
  static const char *const P2[] = { "->", "++", "--", "<<", ">>", "<=", ">=", "==", "!=", "&&",
                                    "||", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "##", NULL };
  int depth = 0, opened = 0, bol = 1;
  size_t i = 0;
  while (i < n) {
    char ch = m[i];
    if (ch == '\n') { bol = 1; i++; continue; }
    if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\f' || ch == '\v') { i++; continue; }
    if (bol && ch == '#') { while (i < n && m[i] != '\n') i++; continue; }   /* #line */
    bol = 0;
    if (ch == '\\' && i + 1 < n && m[i + 1] == '\n') { i += 2; continue; }
    if (gl_idch(ch) && !isdigit((unsigned char)ch)) {
      size_t s = i;
      while (i < n && gl_idch(m[i])) i++;
      gl_tok(F, GT_ID, s, i - s);
      continue;
    }
    if (isdigit((unsigned char)ch) || (ch == '.' && i + 1 < n && isdigit((unsigned char)m[i + 1]))) {
      size_t s = i++;
      while (i < n && (gl_idch(m[i]) || m[i] == '.' ||
                       ((m[i] == '+' || m[i] == '-') && strchr("eEpP", m[i - 1])))) i++;
      gl_tok(F, GT_NUM, s, i - s);
      continue;
    }
    if (ch == '"' || ch == '\'') {
      char q = ch;
      size_t s = i++;
      while (i < n && m[i] != q && m[i] != '\n') i++;
      if (i < n && m[i] == q) i++;
      gl_tok(F, q == '"' ? GT_STR : GT_CHR, s, i - s);
      continue;
    }
    size_t pl = 1;
    for (int k = 0; P3[k]; k++) if (i + 3 <= n && !strncmp(m + i, P3[k], 3)) { pl = 3; break; }
    if (pl == 1) for (int k = 0; P2[k]; k++) if (i + 2 <= n && !strncmp(m + i, P2[k], 2)) { pl = 2; break; }
    gl_tok(F, GT_P, i, pl);
    if (pl == 1 && ch == '{') { depth++; opened = 1; }
    if (pl == 1 && ch == '}') { depth--; if (opened && depth == 0) return 0; if (depth < 0) return -1; }
    i += pl;
  }
  return -1;
}
static int gl_brackets(GFn *F) {
  F->mt = malloc(sizeof *F->mt * (size_t)(F->nt + 1));
  int *st = malloc(sizeof *st * (size_t)(F->nt + 1));
  if (!F->mt || !st) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  int sp = 0, ok = 1;
  for (int i = 0; i < F->nt; i++) {
    F->mt[i] = -1;
    if (F->t[i].k != GT_P || F->t[i].len != 1) continue;
    char ch = F->m[F->t[i].off];
    if (ch == '(' || ch == '[' || ch == '{') st[sp++] = i;
    else if (ch == ')' || ch == ']' || ch == '}') {
      if (!sp) { ok = 0; break; }
      int o = st[--sp];
      char oc = F->m[F->t[o].off];
      if ((oc == '(') != (ch == ')') || (oc == '[') != (ch == ']')) { ok = 0; break; }
      F->mt[o] = i; F->mt[i] = o;
    }
  }
  if (sp) ok = 0;
  free(st);
  return ok;
}
static int tk_is(GFn *F, int i, const char *s) {
  if (i < 0 || i >= F->nt) return 0;
  size_t n = strlen(s);
  return (size_t)F->t[i].len == n && !memcmp(F->m + F->t[i].off, s, n);
}
static int tk_id(GFn *F, int i) { return i >= 0 && i < F->nt && F->t[i].k == GT_ID; }

/* ---- the graph ---- */

static void gl_bail(GFn *F, const char *why) { if (!F->bail) F->bail = why; }
static int gb_new(GFn *F) {
  if (F->nb == F->cb) {
    F->cb = F->cb ? F->cb * 2 : 64;
    F->b = realloc(F->b, sizeof *F->b * (size_t)F->cb);
    if (!F->b) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  }
  memset(&F->b[F->nb], 0, sizeof F->b[0]);
  return F->nb++;
}
static void gb_edge(GFn *F, int from, int to) {
  if (from < 0 || to < 0) return;
  GBlk *bl = &F->b[from];
  for (int i = 0; i < bl->nsucc; i++) if (bl->succ[i] == to) return;
  if (bl->nsucc == bl->csucc) {
    bl->csucc = bl->csucc ? bl->csucc * 2 : 2;
    bl->succ = realloc(bl->succ, sizeof *bl->succ * (size_t)bl->csucc);
    if (!bl->succ) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  }
  bl->succ[bl->nsucc++] = to;
}
static void ge_push(GFn *F, int k, int v) {
  if (F->cur < 0) F->cur = gb_new(F);   /* code no edge reaches */
  GBlk *bl = &F->b[F->cur];
  if (bl->nev == bl->cev) {
    bl->cev = bl->cev ? bl->cev * 2 : 8;
    bl->ev = realloc(bl->ev, sizeof *bl->ev * (size_t)bl->cev);
    if (!bl->ev) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  }
  bl->ev[bl->nev].k = k; bl->ev[bl->nev].v = v;
  bl->nev++;
}
static void gi_add(GInfo *in, int v) {
  if (!in) return;
  for (int i = 0; i < in->n; i++) if (in->v[i] == v) return;
  if (in->n == in->c) {
    in->c = in->c ? in->c * 2 : 8;
    in->v = realloc(in->v, sizeof *in->v * (size_t)in->c);
    if (!in->v) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  }
  in->v[in->n++] = v;
}
static void gi_merge(GInfo *to, const GInfo *from, int below) {
  if (!to) return;
  for (int i = 0; i < from->n; i++) if (from->v[i] < below) gi_add(to, from->v[i]);
  if (from->col) to->col = 1;
}
static void gl_use(GFn *F, int v) { ge_push(F, GE_USE, v); gi_add(F->ci, v); }
/* A write ends the old value's life, unless the write may not run (a part
   of `a ? b : c`, `a && b`) and the variable was there before that part. */
static void gl_def(GFn *F, int v) {
  if (F->cond > 0 && v < F->cond_mark) { F->v[v].wr |= 2; return; }
  F->v[v].wr |= 1;
  ge_push(F, GE_DEF, v);
}
static void gl_col(GFn *F) { ge_push(F, GE_COL, -1); if (F->ci) F->ci->col = 1; F->ncol++; }

/* ---- variables and scopes ---- */

static int gv_new(GFn *F, int off, int len) {
  if (F->nv == F->cv) {
    F->cv = F->cv ? F->cv * 2 : 64;
    F->v = realloc(F->v, sizeof *F->v * (size_t)F->cv);
    if (!F->v) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  }
  GVar *x = &F->v[F->nv];
  memset(x, 0, sizeof *x);
  x->off = off; x->len = len;
  return F->nv++;
}
static void gv_show(GFn *F, int v) {
  if (F->nsv == F->csv) {
    F->csv = F->csv ? F->csv * 2 : 64;
    F->sv = realloc(F->sv, sizeof *F->sv * (size_t)F->csv);
    if (!F->sv) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  }
  F->sv[F->nsv++] = v;
}
static int gv_find(GFn *F, int off, int len) {
  for (int k = F->nsv - 1; k >= 0; k--) {
    GVar *x = &F->v[F->sv[k]];
    if (x->len == len && !memcmp(F->m + x->off, F->m + off, (size_t)len)) return F->sv[k];
  }
  return -1;
}
static int gv_tok(GFn *F, int i) { return tk_id(F, i) ? gv_find(F, F->t[i].off, F->t[i].len) : -1; }
static void gs_push(GFn *F) {
  if (F->nsm >= (int)(sizeof F->sm / sizeof F->sm[0])) { gl_bail(F, "nesting"); return; }
  F->sm[F->nsm++] = F->nsv;
}
static void gs_pop(GFn *F) { if (F->nsm > 0) F->nsv = F->sm[--F->nsm]; }
static void gv_src(GFn *F, int v, const GInfo *in) {
  GVar *x = &F->v[v];
  for (int i = 0; i < in->n; i++) {
    int s = in->v[i], dup = 0;
    if (s == v) continue;
    for (int k = 0; k < x->nsrc; k++) if (x->src[k] == s) { dup = 1; break; }
    if (dup) continue;
    if (x->nsrc == x->csrc) {
      x->csrc = x->csrc ? x->csrc * 2 : 4;
      x->src = realloc(x->src, sizeof *x->src * (size_t)x->csrc);
      if (!x->src) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
    }
    x->src[x->nsrc++] = s;
  }
}

/* ---- statements ---- */

static void gl_stmt(GFn *F);
static void gl_expr(GFn *F, int lo, int hi);

/* the `;` that ends the statement from i, past balanced brackets */
static int gl_semi(GFn *F, int i, int lim) {
  while (i < lim) {
    if (F->t[i].k == GT_P && F->t[i].len == 1) {
      char ch = F->m[F->t[i].off];
      if (ch == ';') return i;
      if (ch == '(' || ch == '[' || ch == '{') { if (F->mt[i] < 0) return -1; i = F->mt[i] + 1; continue; }
      if (ch == ')' || ch == ']' || ch == '}') return -1;
    }
    i++;
  }
  return -1;
}
/* the first token from i on that is `s` at this bracket depth, before lim */
static int gl_find(GFn *F, int i, int lim, const char *s) {
  while (i < lim) {
    if (tk_is(F, i, s)) return i;
    if (F->t[i].k == GT_P && F->t[i].len == 1 && strchr("([{", F->m[F->t[i].off]) && F->mt[i] > i) { i = F->mt[i] + 1; continue; }
    i++;
  }
  return -1;
}
static int gl_label(GFn *F, int i) {
  for (int k = 0; k < F->nlab; k++)
    if (F->lab[k].len == F->t[i].len && !memcmp(F->m + F->lab[k].off, F->m + F->t[i].off, (size_t)F->t[i].len))
      return F->lab[k].blk;
  if (F->nlab == F->clab) {
    F->clab = F->clab ? F->clab * 2 : 8;
    F->lab = realloc(F->lab, sizeof *F->lab * (size_t)F->clab);
    if (!F->lab) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  }
  F->lab[F->nlab].off = F->t[i].off; F->lab[F->nlab].len = F->t[i].len;
  F->lab[F->nlab].blk = gb_new(F);
  return F->lab[F->nlab++].blk;
}

/* A declaration starts at i: storage and qualifiers, a type (words, a
   struct, a __typeof__), stars, a name, then `=`, `;`, `,` or `[`. */
static int gl_type_end(GFn *F, int i, int *scalar) {
  int k = i, sc = 1;
  while (tk_is(F, k, "static") || tk_is(F, k, "const") || tk_is(F, k, "volatile") ||
         tk_is(F, k, "register") || tk_is(F, k, "extern") || tk_is(F, k, "SP_UNUSED") ||
         tk_is(F, k, "__extension__")) k++;
  if (tk_is(F, k, "struct") || tk_is(F, k, "union") || tk_is(F, k, "enum")) {
    sc = tk_is(F, k, "enum");
    k++;
    if (tk_id(F, k)) k++;
    if (tk_is(F, k, "{")) { if (F->mt[k] < 0) return -1; k = F->mt[k] + 1; }
  }
  else if (tk_is(F, k, "__typeof__") || tk_is(F, k, "__typeof") || tk_is(F, k, "typeof")) {
    if (!tk_is(F, k + 1, "(") || F->mt[k + 1] < 0) return -1;
    k = F->mt[k + 1] + 1;
    sc = 0;
  }
  else {
    int s = k;
    while (tk_id(F, k)) k++;
    if (k == s) return -1;
    /* words then a star: all of them are the type; words alone: the last
       one is the name */
    if (!tk_is(F, k, "*")) { if (k - s < 2) return -1; k--; }
    for (int q = s; q < k; q++) if (!gl_scalar_type(F->m + F->t[q].off, (size_t)F->t[q].len)) sc = 0;
  }
  if (scalar) *scalar = sc;
  return k;
}
static int gl_is_decl(GFn *F, int i) {
  static const char *const KW[] = { "return", "if", "else", "while", "do", "for", "switch", "case",
                                    "default", "break", "continue", "goto", "sizeof", NULL };
  if (!tk_id(F, i) || gl_in(KW, F->m + F->t[i].off, (size_t)F->t[i].len)) return 0;
  int k = gl_type_end(F, i, NULL);
  if (k < 0) return 0;
  while (tk_is(F, k, "*") || tk_is(F, k, "const") || tk_is(F, k, "volatile") ||
         tk_is(F, k, "restrict") || tk_is(F, k, "__restrict")) k++;
  if (!tk_id(F, k)) return 0;
  return tk_is(F, k + 1, "=") || tk_is(F, k + 1, ";") || tk_is(F, k + 1, ",") || tk_is(F, k + 1, "[");
}
static void gl_decl(GFn *F, int lim) {
  int scalar = 1, stat = 0;
  for (int q = F->i; q < lim && tk_id(F, q); q++)
    if (tk_is(F, q, "static") || tk_is(F, q, "extern")) stat = 1;
  int k = gl_type_end(F, F->i, &scalar);
  if (k < 0) { gl_bail(F, "declaration"); return; }
  for (;;) {
    int stars = 0;
    while (tk_is(F, k, "*") || tk_is(F, k, "const") || tk_is(F, k, "volatile") ||
           tk_is(F, k, "restrict") || tk_is(F, k, "__restrict")) { if (tk_is(F, k, "*")) stars++; k++; }
    if (!tk_id(F, k)) { gl_bail(F, "declarator"); return; }
    int name = k++, arr = 0;
    while (tk_is(F, k, "[")) { if (F->mt[k] < 0) { gl_bail(F, "array"); return; } arr = 1; k = F->mt[k] + 1; }
    GInfo in = {0};
    int has_init = 0;
    if (tk_is(F, k, "=")) {
      int lo = k + 1, hi = lo;
      while (hi < lim && !tk_is(F, hi, ",") && !tk_is(F, hi, ";")) {
        if (F->t[hi].k == GT_P && F->t[hi].len == 1 && strchr("([{", F->m[F->t[hi].off]) && F->mt[hi] > hi) hi = F->mt[hi] + 1;
        else hi++;
      }
      GInfo *sv = F->ci; F->ci = &in;
      gl_expr(F, lo, hi);
      F->ci = sv;
      gi_merge(sv, &in, F->nv + 1);
      k = hi; has_init = 1;
    }
    int v = gv_new(F, F->t[name].off, F->t[name].len);
    F->v[v].ptr = (char)(stars || arr || !scalar);
    F->v[v].keep = (char)stat;
    F->v[v].dblk = F->cur; F->v[v].dcol = F->ncol;
    gv_src(F, v, &in);
    free(in.v);
    gv_show(F, v);
    if (has_init) ge_push(F, GE_DEF, v);       /* new here: its write always ends nothing */
    if (tk_is(F, k, ",")) { k++; continue; }
    if (tk_is(F, k, ";")) { F->i = k + 1; return; }
    gl_bail(F, "declaration end");
    return;
  }
}
/* An expression statement: `v = e;` writes v after e; anything else is
   read as an expression. */
static void gl_expr_stmt(GFn *F, int lo, int hi) {
  if (tk_id(F, lo) && tk_is(F, lo + 1, "=") && gl_find(F, lo + 2, hi, ",") < 0) {
    int v = gv_tok(F, lo);
    if (v >= 0) {
      GInfo in = {0};
      GInfo *sv = F->ci; F->ci = &in;
      gl_expr(F, lo + 2, hi);
      F->ci = sv;
      gi_merge(sv, &in, F->nv + 1);
      gv_src(F, v, &in);
      free(in.v);
      gl_def(F, v);
      return;
    }
  }
  gl_expr(F, lo, hi);
}
static void gl_cond(GFn *F, int lp) {
  if (!tk_is(F, lp, "(") || F->mt[lp] < 0) { gl_bail(F, "condition"); return; }
  gl_expr(F, lp + 1, F->mt[lp]);
}
static void gl_stmt(GFn *F) {
  if (F->bail) return;
  int i = F->i;
  if (i >= F->nt) { gl_bail(F, "end"); return; }
  if (tk_is(F, i, "{")) {
    int close = F->mt[i];
    if (close < 0) { gl_bail(F, "brace"); return; }
    gs_push(F);
    F->i = i + 1;
    while (!F->bail && F->i < close) gl_stmt(F);
    if (!F->bail && F->i != close) gl_bail(F, "block");
    F->i = close + 1;
    gs_pop(F);
    return;
  }
  if (tk_is(F, i, ";")) { F->i = i + 1; return; }
  if (tk_id(F, i)) {
    if (tk_is(F, i, "if")) {
      gl_cond(F, i + 1);
      if (F->bail) return;
      int ce = F->cur;
      if (ce < 0) ce = F->cur = gb_new(F);
      int tb = gb_new(F);
      gb_edge(F, ce, tb);
      F->cur = tb; F->i = F->mt[i + 1] + 1;
      gl_stmt(F);
      int te = F->cur;
      if (tk_is(F, F->i, "else")) {
        int eb = gb_new(F);
        gb_edge(F, ce, eb);
        F->cur = eb; F->i++;
        gl_stmt(F);
        int ee = F->cur, j = gb_new(F);
        gb_edge(F, te, j); gb_edge(F, ee, j);
        F->cur = j;
      }
      else {
        int j = gb_new(F);
        gb_edge(F, ce, j); gb_edge(F, te, j);
        F->cur = j;
      }
      return;
    }
    if (tk_is(F, i, "while")) {
      int h = gb_new(F);
      gb_edge(F, F->cur, h);
      F->cur = h;
      gl_cond(F, i + 1);
      if (F->bail) return;
      int ce = F->cur, body = gb_new(F), ex = gb_new(F);
      gb_edge(F, ce, body); gb_edge(F, ce, ex);
      if (F->nbrk >= 256 || F->ncnt >= 256) { gl_bail(F, "loops"); return; }
      F->brk[F->nbrk++] = ex; F->cnt[F->ncnt++] = h;
      F->cur = body; F->i = F->mt[i + 1] + 1;
      gl_stmt(F);
      gb_edge(F, F->cur, h);
      F->nbrk--; F->ncnt--;
      F->cur = ex;
      return;
    }
    if (tk_is(F, i, "do")) {
      int body = gb_new(F);
      gb_edge(F, F->cur, body);
      int cb = gb_new(F), ex = gb_new(F);
      if (F->nbrk >= 256 || F->ncnt >= 256) { gl_bail(F, "loops"); return; }
      F->brk[F->nbrk++] = ex; F->cnt[F->ncnt++] = cb;
      F->cur = body; F->i = i + 1;
      gl_stmt(F);
      gb_edge(F, F->cur, cb);
      F->nbrk--; F->ncnt--;
      if (F->bail) return;
      if (!tk_is(F, F->i, "while")) { gl_bail(F, "do"); return; }
      F->cur = cb;
      gl_cond(F, F->i + 1);
      if (F->bail) return;
      int ce = F->cur;
      gb_edge(F, ce, body); gb_edge(F, ce, ex);
      F->i = F->mt[F->i + 1] + 1;
      if (!tk_is(F, F->i, ";")) { gl_bail(F, "do end"); return; }
      F->i++;
      F->cur = ex;
      return;
    }
    if (tk_is(F, i, "for")) {
      int lp = i + 1, rp = tk_is(F, lp, "(") ? F->mt[lp] : -1;
      if (rp < 0) { gl_bail(F, "for"); return; }
      int s1 = gl_find(F, lp + 1, rp, ";"), s2 = s1 >= 0 ? gl_find(F, s1 + 1, rp, ";") : -1;
      if (s2 < 0) { gl_bail(F, "for clauses"); return; }
      gs_push(F);
      if (s1 > lp + 1) {
        if (gl_is_decl(F, lp + 1)) { F->i = lp + 1; gl_decl(F, s1 + 1); }
        else gl_expr_stmt(F, lp + 1, s1);
      }
      int h = gb_new(F);
      gb_edge(F, F->cur, h);
      F->cur = h;
      if (s2 > s1 + 1) gl_expr(F, s1 + 1, s2);
      int ce = F->cur, body = gb_new(F), step = gb_new(F), ex = gb_new(F);
      gb_edge(F, ce, body); gb_edge(F, ce, ex);
      if (F->nbrk >= 256 || F->ncnt >= 256) { gl_bail(F, "loops"); gs_pop(F); return; }
      F->brk[F->nbrk++] = ex; F->cnt[F->ncnt++] = step;
      F->cur = body; F->i = rp + 1;
      gl_stmt(F);
      gb_edge(F, F->cur, step);
      F->nbrk--; F->ncnt--;
      int after = F->i;
      F->cur = step;
      if (rp > s2 + 1) gl_expr_stmt(F, s2 + 1, rp);
      gb_edge(F, F->cur, h);
      F->cur = ex; F->i = after;
      gs_pop(F);
      return;
    }
    if (tk_is(F, i, "switch")) {
      gl_cond(F, i + 1);
      if (F->bail) return;
      int sh = F->cur;
      if (sh < 0) sh = F->cur = gb_new(F);
      int ex = gb_new(F);
      if (F->nsw >= 128 || F->nbrk >= 256) { gl_bail(F, "switches"); return; }
      F->swh[F->nsw] = sh; F->swdef[F->nsw] = 0; F->nsw++;
      F->brk[F->nbrk++] = ex;
      F->cur = -1; F->i = F->mt[i + 1] + 1;
      gl_stmt(F);
      gb_edge(F, F->cur, ex);
      F->nsw--; F->nbrk--;
      if (!F->swdef[F->nsw]) gb_edge(F, sh, ex);
      F->cur = ex;
      return;
    }
    if (tk_is(F, i, "case") || tk_is(F, i, "default")) {
      if (!F->nsw) { gl_bail(F, "case"); return; }
      int colon = gl_find(F, i + 1, F->nt, ":");
      if (colon < 0) { gl_bail(F, "case label"); return; }
      int nb = gb_new(F);
      gb_edge(F, F->cur, nb); gb_edge(F, F->swh[F->nsw - 1], nb);
      if (tk_is(F, i, "default")) F->swdef[F->nsw - 1] = 1;
      F->cur = nb; F->i = colon + 1;
      return;
    }
    if (F->cond > 0 && (tk_is(F, i, "break") || tk_is(F, i, "continue") || tk_is(F, i, "return") ||
                        tk_is(F, i, "goto"))) {
      /* the graph reads `a ? b : c` as b then c: a jump in b would cut c off */
      gl_bail(F, "a jump inside a conditional expression");
      return;
    }
    if (tk_is(F, i, "break") || tk_is(F, i, "continue")) {
      int brk = tk_is(F, i, "break");
      if (brk ? !F->nbrk : !F->ncnt) { gl_bail(F, "jump target"); return; }
      gb_edge(F, F->cur, brk ? F->brk[F->nbrk - 1] : F->cnt[F->ncnt - 1]);
      F->cur = -1;
      if (!tk_is(F, i + 1, ";")) { gl_bail(F, "jump"); return; }
      F->i = i + 2;
      return;
    }
    if (tk_is(F, i, "return")) {
      int e = gl_semi(F, i + 1, F->nt);
      if (e < 0) { gl_bail(F, "return"); return; }
      if (e > i + 1) gl_expr(F, i + 1, e);
      gb_edge(F, F->cur, F->exitb);
      F->cur = -1; F->i = e + 1;
      return;
    }
    if (tk_is(F, i, "goto")) {
      if (!tk_id(F, i + 1) || !tk_is(F, i + 2, ";")) { gl_bail(F, "goto"); return; }
      gb_edge(F, F->cur, gl_label(F, i + 1));
      F->cur = -1; F->i = i + 3;
      return;
    }
    if (tk_is(F, i + 1, ":")) {                 /* a label */
      int lb = gl_label(F, i);
      gb_edge(F, F->cur, lb);
      F->cur = lb; F->i = i + 2;
      return;
    }
    if (tk_is(F, i, "SP_GC_SAVE") && tk_is(F, i + 1, "(") && tk_is(F, i + 2, ")") && tk_is(F, i + 3, ";")) {
      F->i = i + 4;
      return;
    }
    if (gc_leaf_root_stmt(F->m + F->t[i].off, (size_t)F->t[i].len) && tk_is(F, i + 1, "(")) {
      int e = gl_semi(F, i + 1, F->nt);
      if (e < 0) { gl_bail(F, "root"); return; }
      /* `NAME(v);` with v a variable of this function; any other form stays */
      if (!tk_is(F, i, "SP_GC_ROOT_SRANGE") && e == i + 4 && tk_id(F, i + 2) && tk_is(F, i + 3, ")")) {
        int v = gv_tok(F, i + 2);
        if (v >= 0) {
          /* a root that follows the declaration with no collection between
             covers every value the variable holds; a later one does not
             cover what it held before, so its writes' sources stay in play */
          int tight = F->cur >= 0 && F->cur == F->v[v].dblk && F->ncol == F->v[v].dcol;
          if (F->v[v].rooted != 1) F->v[v].rooted = (char)(tight ? 1 : 2);
          /* a parameter rooted before anything can collect, on every path
             from the entry, is safe from the caller's side for the whole
             call, if nothing writes it (gl_written gave the candidates) */
          if (F->v[v].par && tight)
            for (int k = 0; k < F->np; k++) {
              if (F->pvar[k] != v) continue;
              F->ent[k] = 1;
              if (F->cons[k] == 2) {
                F->cons[k] = 1;
                if (F->self_ce >= 0) g_cal[F->self_ce].cons[k] = 1;
              }
            }
          if (F->nr == F->cr) {
            F->cr = F->cr ? F->cr * 2 : 16;
            F->r = realloc(F->r, sizeof *F->r * (size_t)F->cr);
            if (!F->r) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
          }
          F->r[F->nr].beg = i; F->r[F->nr].end = e; F->r[F->nr].var = v;
          F->nr++;
        }
      }
      F->i = e + 1;
      return;
    }
    if (gl_is_decl(F, i)) {
      int e = gl_semi(F, i, F->nt);
      if (e < 0) { gl_bail(F, "declaration"); return; }
      gl_decl(F, e + 1);
      return;
    }
  }
  int e = gl_semi(F, i, F->nt);
  if (e < 0) { gl_bail(F, "statement"); return; }
  gl_expr_stmt(F, i, e);
  F->i = e + 1;
}

/* ---- expressions ---- */

/* the C arguments of the call whose `(` is at lp */
static int gl_args(GFn *F, int lp, int *lo, int *hi, int cap) {
  int rp = F->mt[lp], n = 0, s = lp + 1;
  if (rp == lp + 1) return 0;
  for (int k = lp + 1; k <= rp && n < cap; ) {
    if (k == rp || tk_is(F, k, ",")) { lo[n] = s; hi[n] = k; n++; s = k + 1; k++; continue; }
    if (F->t[k].k == GT_P && F->t[k].len == 1 && strchr("([{", F->m[F->t[k].off]) && F->mt[k] > k) { k = F->mt[k] + 1; continue; }
    k++;
  }
  return n;
}
enum { GK_PURE, GK_METHOD, GK_ROOTS, GK_OTHER };
/* A call: name at ni (-1 for a call through a pointer), `(` at lp. */
static void gl_call(GFn *F, int ni, int lp) {
  GInfo *cc = F->ci;
  int kind = GK_OTHER;
  GCallee *ce = NULL;
  if (ni >= 0) {
    const char *nm = F->m + F->t[ni].off;
    size_t nn = (size_t)F->t[ni].len;
    int fv = gv_tok(F, ni);
    if (fv >= 0) gl_use(F, fv);               /* a call through a variable */
    else if (gc_leaf_call_ok(nm, nn) || gl_noreturn(nm, nn) || gl_pure_extra(nm, nn)) kind = GK_PURE;
    else if (gl_roots_args(nm, nn)) kind = GK_ROOTS;
    else if ((ce = gl_callee(nm, nn)) != NULL) {
      kind = ce->leaf ? GK_PURE : GK_METHOD;
      if (F->self_ce >= 0 && ce == &g_cal[F->self_ce]) F->selfcall = 1;
    }
  }
  int mac = ni >= 0 && gl_macro(F->m + F->t[ni].off, (size_t)F->t[ni].len);
  enum { MAXA = 64 };
  int lo[MAXA], hi[MAXA];
  int na = gl_args(F, lp, lo, hi, MAXA);
  if (na == MAXA) { gl_bail(F, "arguments"); return; }
  GInfo ai[MAXA];
  memset(ai, 0, sizeof ai);
  for (int k = 0; k < na && !F->bail; k++) {
    F->ci = &ai[k];
    gl_expr(F, lo[k], hi[k]);
  }
  F->ci = cc;
  /* the arguments are siblings: one evaluated before a later one's
     collection can be read after it */
  int last = -1;
  for (int k = 0; k < na; k++) if (ai[k].col) last = k;
  for (int k = 0; k < last; k++) for (int q = 0; q < ai[k].n; q++) gl_use(F, ai[k].v[q]);
  if (kind != GK_PURE) gl_col(F);
  for (int k = 0; k < na; k++) {
    int after = kind == GK_OTHER || (kind == GK_METHOD && !(k < ce->np && ce->cons[k] == 1));
    if (after) for (int q = 0; q < ai[k].n; q++) gl_use(F, ai[k].v[q]);
  }
  /* a macro may assign a variable it is given from the other arguments */
  if (mac)
    for (int k = 0; k < na; k++) {
      if (hi[k] != lo[k] + 1) continue;
      int v = gv_tok(F, lo[k]);
      if (v < 0) continue;
      F->v[v].wr |= 2;
      for (int j = 0; j < na; j++) if (j != k) gv_src(F, v, &ai[j]);
    }
  for (int k = 0; k < na; k++) { gi_merge(cc, &ai[k], F->nv + 1); free(ai[k].v); }
}
static void gl_stmtexpr(GFn *F, int open) {
  int close = F->mt[open], mark = F->nv;
  GInfo inner = {0};
  GInfo *cc = F->ci;
  F->ci = &inner;
  gs_push(F);
  int save = F->i;
  F->i = open + 1;
  while (!F->bail && F->i < close) gl_stmt(F);
  if (!F->bail && F->i != close) gl_bail(F, "statement expression");
  F->i = save;
  gs_pop(F);
  F->ci = cc;
  gi_merge(cc, &inner, mark);              /* its own variables stay inside */
  free(inner.v);
}
/* (lp..rp) is a cast: type words and stars, the first word no variable
   of this function. `(void *)(x)` groups x, it calls nothing. */
static int gl_cast(GFn *F, int lp, int rp) {
  if (lp < 0 || rp <= lp + 1 || !tk_id(F, lp + 1) || gv_tok(F, lp + 1) >= 0) return 0;
  for (int k = lp + 1; k < rp; k++) if (!tk_id(F, k) && !tk_is(F, k, "*")) return 0;
  return 1;
}
static int gl_assign_op(GFn *F, int k) {
  static const char *const OPS[] = { "=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", NULL };
  if (k < 0 || k >= F->nt || F->t[k].k != GT_P) return 0;
  return gl_in(OPS, F->m + F->t[k].off, (size_t)F->t[k].len);
}
/* A write inside an expression, to v or into it: `(v = e)`, `v[i] = e`,
   `v.f = e`, `v += e`, `v++`. v may be written, and holds what e reads.
   The variables come from the text of e, which can name more than e
   reads: more sources keep more alive. No DEF: the old value may stay. */
static void gl_store(GFn *F, int v, int i, int hi) {
  if (tk_is(F, i - 1, "++") || tk_is(F, i - 1, "--")) { F->v[v].wr |= 2; return; }
  int k = i + 1;
  for (;;) {
    if (tk_is(F, k, ".") && tk_id(F, k + 1)) { k += 2; continue; }
    if (tk_is(F, k, "[") && F->mt[k] > k) { k = F->mt[k] + 1; continue; }
    break;
  }
  if (tk_is(F, k, "++") || tk_is(F, k, "--")) { F->v[v].wr |= 2; return; }
  if (!gl_assign_op(F, k)) return;
  F->v[v].wr |= 2;
  GInfo in = {0};
  for (int q = k + 1; q < hi; q++) {
    if (tk_is(F, q - 1, ".") || tk_is(F, q - 1, "->")) continue;
    int s = gv_tok(F, q);
    if (s >= 0) gi_add(&in, s);
  }
  gv_src(F, v, &in);
  free(in.v);
}
/* An expression: its atoms in the order of the text, then, since C does
   not fix the order of siblings, a read of each atom's variables after the
   last atom that can collect. Past the first `&&`, `||` or `?` the rest
   may not run: a write there ends no older variable's value (gl_def). */
static void gl_expr(GFn *F, int lo, int hi) {
  GInfo *outer = F->ci;
  GInfo *at = NULL; int na = 0, ca = 0;
  int i = lo;
  int cond_on = 0, save_mark = F->cond_mark;
  while (i < hi && !F->bail) {
    if (!cond_on && F->t[i].k == GT_P && (tk_is(F, i, "&&") || tk_is(F, i, "||") || tk_is(F, i, "?"))) {
      cond_on = 1; F->cond++; F->cond_mark = F->nv;
    }
    int j = i + 1, atom = 0;
    GInfo cur = {0};
    F->ci = &cur;
    if (F->t[i].k == GT_P && F->t[i].len == 1) {
      char ch = F->m[F->t[i].off];
      if (ch == '(') {
        int r = F->mt[i];
        if (r < 0 || r >= hi) { gl_bail(F, "paren"); break; }
        if (tk_is(F, i + 1, "{") && F->mt[i + 1] == r - 1) gl_stmtexpr(F, i + 1);
        else if (i > lo && (tk_is(F, i - 1, "]") || F->t[i - 1].k == GT_ID ||
                            (tk_is(F, i - 1, ")") && !gl_cast(F, F->mt[i - 1], i - 1)))) gl_call(F, -1, i);
        else gl_expr(F, i + 1, r);
        atom = 1; j = r + 1;
      }
      else if (ch == '[' || ch == '{') {
        int r = F->mt[i];
        if (r < 0 || r >= hi) { gl_bail(F, "bracket"); break; }
        gl_expr(F, i + 1, r);
        atom = 1; j = r + 1;
      }
      else if (ch == '&' && !(i > lo && (F->t[i - 1].k == GT_ID || F->t[i - 1].k == GT_NUM ||
                                         tk_is(F, i - 1, ")") || tk_is(F, i - 1, "]")))) {
        /* `&v`, `&(v)`, `&v.f`, `&v[i]`: v's address escapes. `&v->f` is
           inside what v points to, and takes nothing of v */
        int q = i + 1, o = 0;
        while (tk_is(F, q, "(")) { q++; o++; }
        int v = gv_tok(F, q), k = q + 1;
        while (o > 0 && tk_is(F, k, ")")) { k++; o--; }
        if (v >= 0 && !tk_is(F, k, "->")) { F->v[v].addr = 1; F->v[v].wr |= 2; }
      }
    }
    else if (F->t[i].k == GT_ID) {
      if (tk_is(F, i - 1, ".") || tk_is(F, i - 1, "->")) { /* a field */ }
      else if (tk_is(F, i + 1, "(")) {
        int r = F->mt[i + 1];
        if (r < 0 || r >= hi) { gl_bail(F, "call"); break; }
        if (!gl_uneval(F->m + F->t[i].off, (size_t)F->t[i].len)) { gl_call(F, i, i + 1); atom = 1; }
        j = r + 1;
      }
      else {
        int v = gv_tok(F, i);
        if (v >= 0) { gl_use(F, v); atom = 1; gl_store(F, v, i, hi); }
      }
    }
    F->ci = outer;
    if (atom && (cur.n || cur.col)) {
      if (na == ca) {
        ca = ca ? ca * 2 : 8;
        at = realloc(at, sizeof *at * (size_t)ca);
        if (!at) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
      }
      at[na++] = cur;
    }
    else free(cur.v);
    i = j;
  }
  F->ci = outer;
  if (cond_on) { F->cond--; F->cond_mark = save_mark; }
  int last = -1;
  for (int k = 0; k < na; k++) if (at[k].col) last = k;
  for (int k = 0; k < last; k++) for (int q = 0; q < at[k].n; q++) gl_use(F, at[k].v[q]);
  for (int k = 0; k < na; k++) { gi_merge(outer, &at[k], F->nv + 1); free(at[k].v); }
  free(at);
}

/* ---- liveness ---- */

/* A variable Ruby code names: a local or parameter (lv_), a captured one
   (_cell_), self. Its object lives as long as the name, as in CRuby. */
static int gl_ruby_named(GFn *F, int v) {
  const char *nm = F->m + F->v[v].off;
  int n = F->v[v].len;
  return (n > 3 && !strncmp(nm, "lv_", 3)) || (n > 6 && !strncmp(nm, "_cell_", 6)) ||
         (n == 4 && !strncmp(nm, "self", 4));
}
/* Sets each variable's need (live just after some collection point) and
   each parameter's leak (a whole write of it leaves a variable live whose
   value can come from the old one, through no root of its own). */
static void gl_live(GFn *F) {
  int nv = F->nv, nb = F->nb, W = (nv + 63) / 64;
  if (W == 0) return;
  if ((size_t)nb * (size_t)W > ((size_t)1 << 22)) { gl_bail(F, "size"); return; }
  uint64_t *in = calloc((size_t)nb * (size_t)W, sizeof *in);
  uint64_t *out = calloc((size_t)nb * (size_t)W, sizeof *out);
  uint64_t *live = calloc((size_t)W, sizeof *live), *need = calloc((size_t)W, sizeof *need);
  /* what a read of each variable keeps alive: itself, and for a pointer
     with no root right after its declaration, the variables its writes
     read, through further such ones */
  uint64_t **P = calloc((size_t)nv, sizeof *P);
  int *stk = malloc(sizeof *stk * (size_t)(nv + 1));
  if (!in || !out || !live || !need || !P || !stk) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  for (int v = 0; v < nv; v++) {
    GVar *x = &F->v[v];
    if (x->rooted == 1 || !x->ptr || !x->nsrc) continue;
    P[v] = calloc((size_t)W, sizeof *P[v]);
    if (!P[v]) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
    int sp = 0;
    stk[sp++] = v; P[v][v >> 6] |= 1ull << (v & 63);
    while (sp) {
      int u = stk[--sp];
      for (int k = 0; k < F->v[u].nsrc; k++) {
        int s = F->v[u].src[k];
        if (!F->v[s].ptr || (P[v][s >> 6] >> (s & 63) & 1)) continue;
        P[v][s >> 6] |= 1ull << (s & 63);
        if (F->v[s].rooted != 1) stk[sp++] = s;
      }
    }
  }
  int changed = 1, rounds = 0;
  while (changed) {
    changed = 0;
    if (++rounds > 1000) { gl_bail(F, "rounds"); break; }
    for (int bi = nb - 1; bi >= 0; bi--) {
      GBlk *bl = &F->b[bi];
      uint64_t *o = out + (size_t)bi * (size_t)W;
      for (int s = 0; s < bl->nsucc; s++) {
        const uint64_t *si = in + (size_t)bl->succ[s] * (size_t)W;
        for (int w = 0; w < W; w++) o[w] |= si[w];
      }
      memcpy(live, o, sizeof *live * (size_t)W);
      for (int e = bl->nev - 1; e >= 0; e--) {
        int v = bl->ev[e].v;
        if (bl->ev[e].k == GE_USE) {
          if (P[v]) for (int w = 0; w < W; w++) live[w] |= P[v][w];
          else live[v >> 6] |= 1ull << (v & 63);
        }
        else if (bl->ev[e].k == GE_DEF) live[v >> 6] &= ~(1ull << (v & 63));
      }
      uint64_t *ii = in + (size_t)bi * (size_t)W;
      if (memcmp(ii, live, sizeof *live * (size_t)W)) { memcpy(ii, live, sizeof *live * (size_t)W); changed = 1; }
    }
  }
  if (!F->bail) {
    for (int bi = 0; bi < nb; bi++) {
      GBlk *bl = &F->b[bi];
      memcpy(live, out + (size_t)bi * (size_t)W, sizeof *live * (size_t)W);
      for (int e = bl->nev - 1; e >= 0; e--) {
        int v = bl->ev[e].v;
        if (bl->ev[e].k == GE_COL) for (int w = 0; w < W; w++) need[w] |= live[w];
        else if (bl->ev[e].k == GE_USE) {
          if (P[v]) for (int w = 0; w < W; w++) live[w] |= P[v][w];
          else live[v >> 6] |= 1ull << (v & 63);
        }
        else {
          if (F->v[v].par)
            for (int k = 0; k < F->np; k++) {
              if (F->pvar[k] != v) continue;
              for (int u = 0; u < nv && !F->leak[k]; u++)
                if (u != v && P[u] && (live[u >> 6] >> (u & 63) & 1) && (P[u][v >> 6] >> (v & 63) & 1))
                  F->leak[k] = 1;
            }
          live[v >> 6] &= ~(1ull << (v & 63));
        }
      }
    }
    for (int v = 0; v < nv; v++) {
      F->v[v].need = (char)(need[v >> 6] >> (v & 63) & 1);
      if (F->ncol > 0 && gl_ruby_named(F, v)) F->v[v].need = 1;
    }
  }
  for (int v = 0; v < nv; v++) free(P[v]);
  free(P); free(stk); free(in); free(out); free(live); free(need);
}

/* Is the root statement ending at token e the last statement of a statement
   expression, `({ ...; sp_RbVal v = e; SP_GC_ROOT_RBVAL(v); })`? The frame
   build turns that declaration into a store to v's frame slot, and the store
   gives the expression its value. Without the root, the declaration stays,
   and the expression has no value: the C does not compile. */
static int gl_ends_stmtexpr(GFn *F, int e) {
  int j = e + 1;
  while (tk_is(F, j, ";")) j++;
  if (!tk_is(F, j, "}")) return 0;
  int open = F->mt[j];
  return open > 0 && tk_is(F, open - 1, "(");
}

/* Takes back the roots whose variable no collection point needs, last
   first so the offsets before them hold. */
static int gl_drop(GFn *F, const char *fname, const char *site, Buf *b, size_t base) {
  int dropped = 0, kept = 0;
  for (int k = F->nr - 1; k >= 0; k--) {
    GRoot *rt = &F->r[k];
    GVar *x = &F->v[rt->var];
    char nm[128];
    snprintf(nm, sizeof nm, "%.*s", x->len, F->m + x->off);
    int last = gl_ends_stmtexpr(F, rt->end);
    if (x->need || x->addr || x->keep || last || !decide_fn("root-live", site, nm)) {
      kept++;
      if (getenv("SP_ROOT_LIVE_LOG"))
        fprintf(stderr, "root-live: %s keeps %s (%s)\n", fname, nm,
                x->need ? "read after a collection" : x->addr ? "address taken" : x->keep ? "static" :
                last ? "ends a statement expression" : "decision");
      continue;
    }
    size_t beg = base + (size_t)F->t[rt->beg].off, end = base + (size_t)F->t[rt->end].off + 1;
    /* a statement alone on its line goes with its line */
    size_t ls = beg, le = end;
    while (ls > 0 && (b->p[ls - 1] == ' ' || b->p[ls - 1] == '\t')) ls--;
    while (le < b->len && (b->p[le] == ' ' || b->p[le] == '\t')) le++;
    if ((ls == 0 || b->p[ls - 1] == '\n') && le < b->len && b->p[le] == '\n') { beg = ls; end = le + 1; }
    else { while (end < b->len && b->p[end] == ' ') end++; }
    buf_erase(b, beg, end - beg);
    dropped++;
    if (getenv("SP_ROOT_LIVE_LOG")) fprintf(stderr, "root-live: %s drops %s\n", fname, nm);
  }
  if (getenv("SP_ROOT_LIVE_LOG"))
    fprintf(stderr, "root-live: %s: %d blocks, %d variables, %d collection points, %d roots dropped, %d kept\n",
            fname, F->nb, F->nv, F->ncol, dropped, kept);
  return dropped;
}

/* the function's signature: `name(` from the start of its line, before from */
static size_t gl_sig(const Buf *b, size_t from, const char *fname) {
  size_t n = strlen(fname);
  size_t lim = from > ((size_t)1 << 20) ? from - ((size_t)1 << 20) : 0;
  for (size_t i = from; i-- > lim; ) {
    if (b->p[i] != fname[0] || i + n + 1 > b->len) continue;
    if (strncmp(b->p + i, fname, n) || b->p[i + n] != '(') continue;
    if (i > 0 && gl_idch(b->p[i - 1])) continue;
    size_t ls = i;
    while (ls > 0 && b->p[ls - 1] != '\n') ls--;
    return ls;
  }
  return (size_t)-1;
}

/* Can the body [lo, hi) write the variable named by the token at nm? Read
   off the text, over every mention that is not a field: an assignment to it
   or into it (`x =`, `x += 1`, `x[i] =`, `x.f =`), ++ or --, its address,
   or a mention inside a macro's parentheses, which may assign it. An inner
   variable of the same name counts too. */
static int gl_written(GFn *F, int lo, int hi, int nm) {
  int open[1024], mac[1024], d = 0, inmac = 0;
  for (int i = lo; i < hi; i++) {
    if (F->t[i].k == GT_P && F->t[i].len == 1 && F->m[F->t[i].off] == '(') {
      if (d == 1024) return 1;
      int m = i > lo && tk_id(F, i - 1) && gl_macro(F->m + F->t[i - 1].off, (size_t)F->t[i - 1].len) &&
              !gc_leaf_root_stmt(F->m + F->t[i - 1].off, (size_t)F->t[i - 1].len);
      open[d] = i; mac[d] = m; d++; inmac += m;
      continue;
    }
    if (F->t[i].k == GT_P && F->t[i].len == 1 && F->m[F->t[i].off] == ')') {
      if (d > 0) { d--; inmac -= mac[d]; }
      continue;
    }
    if (!tk_id(F, i) || F->t[i].len != F->t[nm].len ||
        memcmp(F->m + F->t[i].off, F->m + F->t[nm].off, (size_t)F->t[nm].len)) continue;
    if (tk_is(F, i - 1, ".") || tk_is(F, i - 1, "->")) continue;
    if (inmac) return 1;
    if (tk_is(F, i - 1, "&") || tk_is(F, i - 1, "++") || tk_is(F, i - 1, "--")) return 1;
    int k = i + 1;
    for (;;) {
      if (tk_is(F, k, ".") && tk_id(F, k + 1)) { k += 2; continue; }
      if (tk_is(F, k, "[") && F->mt[k] > k) { k = F->mt[k] + 1; continue; }
      break;
    }
    if (gl_assign_op(F, k) || tk_is(F, k, "++") || tk_is(F, k, "--")) return 1;
  }
  (void)open;
  return 0;
}

/* Reads the body into the graph, from scratch. prov[k] says what a call
   of the function from its own body may take for parameter k (1: safe). */
static void gl_read(GFn *F, int name, int body, const char *prov) {
  for (int i = 0; i < F->nb; i++) { free(F->b[i].ev); free(F->b[i].succ); }
  for (int i = 0; i < F->nv; i++) free(F->v[i].src);
  F->nb = F->nv = F->nsv = F->nsm = F->nbrk = F->ncnt = F->nsw = F->nlab = F->nr = 0;
  F->ci = NULL; F->ncol = 0; F->cur = -1; F->cond = 0; F->cond_mark = 0; F->selfcall = 0;
  int lo[64], hi[64];
  int np = gl_args(F, name + 1, lo, hi, 64);
  if (np != F->np) { gl_bail(F, "parameters"); return; }
  F->exitb = gb_new(F);
  F->entryb = gb_new(F);
  gs_push(F);
  for (int k = 0; k < np; k++) {
    F->pvar[k] = -1; F->ent[k] = 0; F->leak[k] = 0; F->cons[k] = 0;
    int last = -1, stars = 0, scalar = 1;
    for (int q = lo[k]; q < hi[k]; q++) {
      if (tk_id(F, q)) last = q;
      if (tk_is(F, q, "*") || tk_is(F, q, "[")) stars = 1;
    }
    if (last < 0 || (hi[k] - lo[k] == 1 && tk_is(F, lo[k], "void"))) continue;
    for (int q = lo[k]; q < last; q++)
      if (tk_id(F, q) && !gl_scalar_type(F->m + F->t[q].off, (size_t)F->t[q].len)) scalar = 0;
    int v = gv_new(F, F->t[last].off, F->t[last].len);
    F->v[v].par = 1;
    F->v[v].ptr = (char)(stars || !scalar);
    F->v[v].dblk = F->entryb; F->v[v].dcol = 0;
    gv_show(F, v);
    F->pvar[k] = v;
    /* what a call from the body may take: nothing to keep for a parameter
       that holds no reference, what the last reading found, and for one
       nothing writes, its root at the entry (2: waits for the root) */
    if (!F->v[v].ptr || (prov && prov[k] == 1)) F->cons[k] = 1;
    else if (!gl_written(F, body, F->nt, last)) F->cons[k] = 2;
  }
  if (F->self_ce >= 0) for (int k = 0; k < np; k++) g_cal[F->self_ce].cons[k] = (char)(F->cons[k] == 1);
  F->cur = F->entryb;
  F->i = body;
  gl_stmt(F);
  gb_edge(F, F->cur, F->exitb);
  if (!F->bail && F->i != F->nt) gl_bail(F, "trailing");
  for (int v = 0; v < F->nv && !F->bail; v++)
    if (F->v[v].addr && F->v[v].ptr && !F->v[v].rooted) gl_bail(F, "address of an unrooted pointer");
}
/* What a caller's argument for parameter k needs after the call starts,
   from a finished reading: nothing (1) when the parameter holds no
   reference, or when it is rooted first thing and only whole writes change
   it with no copy of the old value left. */
static void gl_safe_args(GFn *F, char *out) {
  for (int k = 0; k < F->np; k++) {
    int v = F->pvar[k];
    out[k] = 0;
    if (v < 0 || F->bail) continue;
    GVar *x = &F->v[v];
    if (!x->ptr) out[k] = 1;
    else if (F->ent[k] && !(x->wr & 2) && !F->leak[k]) out[k] = 1;
  }
}

void gc_live_take_back(Compiler *c, Buf *b, size_t from, const char *fname, const char *site) {
  (void)c;
  static int off = -1;
  if (off < 0) off = getenv("SP_NO_ROOT_LIVE") != NULL;
  if (off || from > b->len || !fname) return;
  int logit = getenv("SP_ROOT_LIVE_LOG") != NULL;
  size_t sig = gl_sig(b, from, fname);
  if (sig == (size_t)-1) {
    if (logit) fprintf(stderr, "root-live: %s: no signature\n", fname);
    return;
  }
  char *m = gc_leaf_code_only(b->p, sig, b->len);
  GFn F;
  memset(&F, 0, sizeof F);
  F.m = m;
  F.cur = -1;
  F.self_ce = -1;
  int dup = 0, is_main = !strcmp(fname, "_sp_main_body");
  if (gl_tokenize(&F, m, b->len - sig) < 0 || !gl_brackets(&F)) F.bail = "tokens";
  int name = -1, body = -1;
  if (!F.bail) {
    for (int i = 0; i < F.nt; i++)
      if (tk_id(&F, i) && (size_t)F.t[i].len == strlen(fname) && !memcmp(m + F.t[i].off, fname, strlen(fname)) &&
          tk_is(&F, i + 1, "(")) { name = i; break; }
    if (name < 0 || F.mt[name + 1] < 0) F.bail = "signature";
    else {
      body = F.mt[name + 1] + 1;
      while (body < F.nt && !tk_is(&F, body, "{")) body++;
      /* the body found is the one that holds `from` */
      if (body >= F.nt || F.mt[body] != F.nt - 1 || sig + (size_t)F.t[body].off >= from ||
          sig + (size_t)F.t[F.nt - 1].off < from) F.bail = "body";
    }
  }
  for (int i = 0; i < F.nt && !F.bail; i++)
    if (tk_id(&F, i) && gl_bail_id(m + F.t[i].off, (size_t)F.t[i].len)) F.bail = "setjmp or the root depth";
  char *prov = NULL, *safe = NULL;
  if (!F.bail) {
    int lo[64], hi[64];
    F.np = gl_args(&F, name + 1, lo, hi, 64);
    if (F.np == 64) F.bail = "parameters";
  }
  int np = F.bail ? 0 : F.np;
  size_t pn = (size_t)(np > 0 ? np : 1);
  F.cons = calloc(pn, 1); F.ent = calloc(pn, 1); F.leak = calloc(pn, 1);
  F.pvar = calloc(pn, sizeof *F.pvar);
  prov = calloc(pn, 1); safe = calloc(pn, 1);
  if (!F.cons || !F.ent || !F.leak || !F.pvar || !prov || !safe) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  if (!F.bail && !is_main) {
    F.self_ce = gl_callee_open(fname, np);
    dup = F.self_ce < 0;
  }
  /* a recursion reads the function's own entry: read again with what the
     last reading found, until it finds nothing new (more safe arguments
     can only free more) */
  int solved = 0;
  for (int round = 0; !F.bail && round < 4; round++) {
    gl_read(&F, name, body, round ? prov : NULL);
    if (F.bail) break;
    if (!F.nr) break;
    gl_live(&F);
    if (F.bail) break;
    solved = 1;
    gl_safe_args(&F, safe);
    int more = 0, less = 0;
    for (int k = 0; k < np; k++) {
      if (safe[k] && F.cons[k] != 1) more = 1;
      if (!safe[k] && F.cons[k] == 1) less = 1;
    }
    /* a self-call took cons[k]; what it took must hold */
    if (less && F.selfcall) { gl_bail(&F, "recursion"); solved = 0; break; }
    if (!more || !F.selfcall || round == 3) break;
    memcpy(prov, safe, (size_t)np);
    solved = 0;
  }
  if (!F.bail && solved && F.nr) gl_drop(&F, fname, site, b, sig);
  else if (logit)
    fprintf(stderr, "root-live: %s: %d collection points, %d roots%s%s\n", fname, F.ncol, F.nr,
            F.bail ? ", gave up: " : "", F.bail ? F.bail : "");
  /* what callers read off this function from here on */
  if (!is_main && !dup) {
    if (F.self_ce < 0) F.self_ce = gl_callee_open(fname, 0);
    if (F.self_ce >= 0) {
      GCallee *e = &g_cal[F.self_ce];
      e->leaf = (char)(!F.bail && F.ncol == 0);
      if (F.bail) e->np = 0;
      for (int k = 0; k < e->np; k++) e->cons[k] = (char)(!F.bail && (solved ? safe[k] : F.cons[k] == 1));
      if (logit && e->np) {
        fprintf(stderr, "root-live: %s%s, safe arguments:", fname, e->leaf ? " (no collection)" : "");
        for (int k = 0; k < e->np; k++) fprintf(stderr, " %d", e->cons[k]);
        fputc('\n', stderr);
      }
    }
  }
  free(prov); free(safe); free(F.cons); free(F.ent); free(F.leak); free(F.pvar);
  for (int i = 0; i < F.nb; i++) { free(F.b[i].ev); free(F.b[i].succ); }
  for (int i = 0; i < F.nv; i++) free(F.v[i].src);
  free(F.b); free(F.v); free(F.sv); free(F.t); free(F.mt); free(F.lab); free(F.r);
  free(m);
}
