/* analyze_cfg.c -- a per-method control-flow graph and one forward dataflow
   pass over it (prototype, see SP_NO_CFG).

   A method body is cut into basic blocks: runs of events (a local read, a
   local write) with no jump inside. Edges follow if/unless/case, and/or,
   while/until/for (with their back edges), begin/rescue/ensure (every
   protected block reaches each rescue arm), return/break/next/redo/retry and
   the diverging Kernel calls (raise, exit ...). A block, lambda, def or class
   body is a scope of its own and is not entered: a local a closure writes is
   left out of the facts altogether.

   Two facts per local, kept per block entry and settled by a worklist:
   - unset: the read may run before any write of the local (OR at a join),
   - ty:    the one kind every write that reaches the read stored, or
            BOTTOM when two kinds (or nil, or an unknown write) meet.

   Readers: du_read_maybe_unset (the definite-assignment walk) asks the unset
   fact first; cfg_narrow_reads marks a read of a poly local whose reaching
   writes all stored one unboxable kind in c->nilnarrow, the read-site unbox
   the nil guards and is_a? guards use. */
#include "analyze_internal.h"
#include "builtin_names.h"
#include "repr.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { EV_READ = 1, EV_WRITE, EV_WRITE_UNK, EV_ANDWRITE, EV_OPWRITE };

typedef struct {
  int *ev_kind, *ev_node; int nev, cev;
  int *succ; int nsucc, csucc;
  int reached;
  int *ev_def;       /* SSA mode: the definition a write event makes */
} CBlock;

typedef struct {
  Compiler *c;
  const NodeTable *nt;
  Scope *s;
  int si;
  CBlock *b; int nb, cb;
  int cur;           /* the block events go to; -1 after a jump (dead) */
  int exit_blk;
  /* loop stack */
  int lp_head[64], lp_body[64], lp_exit[64], nlp;
  int lp_endepth[64];
  /* begin stack: entry block of each begin whose rescue arms are being
     lowered (retry re-enters it), and the ensure depth inside it */
  int bg_entry[64], bg_endepth[64], nbg;
  /* ensure stack: a jump out of a protected body runs the ensure first,
     then goes on to its target (routed through every ensure it leaves) */
  int en_block[64], en_nt[64], en_target[64][32], en_tdepth[64][32], nen;
  int protect;       /* >0 inside a body a rescue or ensure guards */
  char *captured;    /* per local index: a closure writes it */
  int want_types;
  /* SSA mode (ssa_rename_locals): locals are known by name, before the
     scope's table exists, and no call is taken to diverge */
  int ssa;
  const char **sn; int nsn;
} CfgB;

#define TY_TOP 0
#define TY_BOT 255

typedef struct {
  unsigned char *ty;   /* per block, per local: TY_TOP, kind+1, TY_BOT */
  unsigned char *unset;
} CState;

static int cfg_disabled(void) {
  static int v = -1;
  if (v < 0) v = getenv("SP_NO_CFG") != NULL;
  return v;
}
static int cfg_log(void) {
  static int v = -1;
  if (v < 0) v = getenv("SP_CFG_LOG") != NULL;
  return v;
}

/* ---- facts, keyed by node id ---- */
static Compiler *fc_c;
static int fc_version = -1, fc_cap;
static unsigned char *fc_unset;  /* 0 set, 1 maybe unset, 2 no fact */
static unsigned char *fc_ty;     /* TyKind the reaching writes stored, else TY_UNKNOWN */
static int st_blocks, st_methods, st_reads, st_unset_set, st_narrowed;

static int cfg_new_block(CfgB *g) {
  if (g->nb >= g->cb) {
    g->cb = g->cb ? g->cb * 2 : 16;
    g->b = realloc(g->b, sizeof(CBlock) * (size_t)g->cb);
    if (!g->b) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  }
  CBlock *bl = &g->b[g->nb];
  memset(bl, 0, sizeof *bl);
  return g->nb++;
}
static void cfg_add_edge(CfgB *g, int from, int to) {
  if (from < 0 || to < 0) return;
  CBlock *bl = &g->b[from];
  for (int i = 0; i < bl->nsucc; i++) if (bl->succ[i] == to) return;
  if (bl->nsucc >= bl->csucc) {
    bl->csucc = bl->csucc ? bl->csucc * 2 : 4;
    bl->succ = realloc(bl->succ, sizeof(int) * (size_t)bl->csucc);
    if (!bl->succ) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  }
  bl->succ[bl->nsucc++] = to;
}
/* a fresh block that follows the current one */
static int cfg_fall(CfgB *g) {
  int nb = cfg_new_block(g);
  if (g->cur >= 0) cfg_add_edge(g, g->cur, nb);
  g->cur = nb;
  return nb;
}
static void cfg_emit(CfgB *g, int kind, int node) {
  /* inside a protected body every write starts a new block, so the state
     before it is a block exit an exception edge carries to the rescue */
  if (g->protect > 0 && kind != EV_READ) cfg_fall(g);
  if (g->cur < 0) g->cur = cfg_new_block(g);   /* dead code: a block no edge reaches */
  CBlock *bl = &g->b[g->cur];
  if (bl->nev >= bl->cev) {
    bl->cev = bl->cev ? bl->cev * 2 : 8;
    bl->ev_kind = realloc(bl->ev_kind, sizeof(int) * (size_t)bl->cev);
    bl->ev_node = realloc(bl->ev_node, sizeof(int) * (size_t)bl->cev);
    if (!bl->ev_kind || !bl->ev_node) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  }
  bl->ev_kind[bl->nev] = kind; bl->ev_node[bl->nev] = node; bl->nev++;
}
static int cfg_local_index(CfgB *g, const char *nm) {
  if (!nm) return -1;
  if (g->ssa) {
    for (int i = 0; i < g->nsn; i++) if (sp_streq(g->sn[i], nm)) return i;
    return -1;
  }
  LocalVar *lv = scope_local(g->s, nm);
  if (!lv) return -1;
  return (int)(lv - g->s->locals);
}
static int cfg_owned(CfgB *g, int n) {
  return comp_scope_of(g->c, n) == g->s;
}

static void cfg_lower(CfgB *g, int n);

static void cfg_lower_list(CfgB *g, int stmts) {
  if (stmts < 0) return;
  int bn = 0; const int *b = nt_arr(g->nt, stmts, "body", &bn);
  for (int i = 0; i < bn; i++) cfg_lower(g, b[i]);
}
/* every LocalVariableTargetNode under a pattern binds when it matches */
static void cfg_lower_pattern_binds(CfgB *g, int n, int depth) {
  if (n < 0 || depth > 64) return;
  const NodeTable *nt = g->nt;
  if (nt_kind(nt, n) == NK_LocalVariableTargetNode) { cfg_emit(g, EV_WRITE_UNK, n); return; }
  for (int i = 0; i < nt_num_refs(nt, n); i++) cfg_lower_pattern_binds(g, nt_ref_at(nt, n, i), depth + 1);
  for (int i = 0; i < nt_num_arrs(nt, n); i++) {
    int an = 0; const int *av = nt_arr_at(nt, n, i, &an);
    for (int k = 0; k < an; k++) cfg_lower_pattern_binds(g, av[k], depth + 1);
  }
}
/* A multiple assignment runs in three steps (`a.b, c[d], e = v, w`): the
   receivers and arguments of its targets, left to right; then the values;
   then the stores, lefts first, then the splat, then rights. */
static void cfg_masgn_operands(CfgB *g, int n, int depth) {
  const NodeTable *nt = g->nt;
  if (n < 0 || depth > 64) return;
  NodeKind k = nt_kind(nt, n);
  if (k == NK_MultiWriteNode || k == NK_MultiTargetNode) {
    int ln = 0; const int *l = nt_arr(nt, n, "lefts", &ln);
    for (int i = 0; i < ln; i++) cfg_masgn_operands(g, l[i], depth + 1);
    cfg_masgn_operands(g, nt_ref(nt, n, "rest"), depth + 1);
    int rn = 0; const int *r = nt_arr(nt, n, "rights", &rn);
    for (int i = 0; i < rn; i++) cfg_masgn_operands(g, r[i], depth + 1);
    return;
  }
  if (k == NK_SplatNode) { cfg_masgn_operands(g, nt_ref(nt, n, "expression"), depth + 1); return; }
  if (k == NK_IndexTargetNode || k == NK_CallTargetNode) {
    cfg_lower(g, nt_ref(nt, n, "receiver"));
    cfg_lower(g, nt_ref(nt, n, "arguments"));
    int blk = nt_ref(nt, n, "block");
    if (blk >= 0 && nt_kind(nt, blk) != NK_BlockNode) cfg_lower(g, blk);
  }
  /* a local, instance, class or global variable target has no operand */
}
static void cfg_masgn_stores(CfgB *g, int n, int depth) {
  const NodeTable *nt = g->nt;
  if (n < 0 || depth > 64) return;
  NodeKind k = nt_kind(nt, n);
  if (k == NK_MultiWriteNode || k == NK_MultiTargetNode) {
    int ln = 0; const int *l = nt_arr(nt, n, "lefts", &ln);
    for (int i = 0; i < ln; i++) cfg_masgn_stores(g, l[i], depth + 1);
    cfg_masgn_stores(g, nt_ref(nt, n, "rest"), depth + 1);
    int rn = 0; const int *r = nt_arr(nt, n, "rights", &rn);
    for (int i = 0; i < rn; i++) cfg_masgn_stores(g, r[i], depth + 1);
    return;
  }
  if (k == NK_SplatNode) { cfg_masgn_stores(g, nt_ref(nt, n, "expression"), depth + 1); return; }
  if (k == NK_LocalVariableTargetNode && cfg_owned(g, n)) cfg_emit(g, EV_WRITE_UNK, n);
  /* the store of any other target writes no local of this method */
}
static void cfg_lower_children(CfgB *g, int n) {
  const NodeTable *nt = g->nt;
  for (int i = 0; i < nt_num_refs(nt, n); i++) cfg_lower(g, nt_ref_at(nt, n, i));
  for (int i = 0; i < nt_num_arrs(nt, n); i++) {
    int an = 0; const int *av = nt_arr_at(nt, n, i, &an);
    for (int k = 0; k < an; k++) cfg_lower(g, av[k]);
  }
}
static void cfg_jump_to(CfgB *g, int target) {
  if (g->cur >= 0) cfg_add_edge(g, g->cur, target);
  g->cur = -1;
}
/* a jump to `target` that leaves every ensure above depth `depth`: it goes
   to the innermost one, which continues it after its body */
static void cfg_jump_out(CfgB *g, int target, int depth) {
  if (g->nen > depth && g->nen <= 64) {
    int f = g->nen - 1;
    if (g->cur >= 0) cfg_add_edge(g, g->cur, g->en_block[f]);
    if (g->en_nt[f] < 32) { g->en_target[f][g->en_nt[f]] = target; g->en_tdepth[f][g->en_nt[f]] = depth; g->en_nt[f]++; }
    else cfg_add_edge(g, g->en_block[f], target);   /* over the limit: a direct edge, still a path */
    g->cur = -1;
    return;
  }
  cfg_jump_to(g, target);
}
/* a two-arm conditional: `then` and `els` (either may be -1) meet after */
static void cfg_lower_arms(CfgB *g, int then, int els, int then_binds) {
  int before = g->cur;
  int join = cfg_new_block(g);
  g->cur = before;
  cfg_fall(g);
  if (then_binds >= 0) cfg_lower_pattern_binds(g, then_binds, 0);
  cfg_lower_list(g, then);
  cfg_jump_to(g, join);
  g->cur = before;
  if (els >= 0) { cfg_fall(g); cfg_lower(g, els); }
  cfg_jump_to(g, join);
  g->cur = join;
}
static void cfg_lower_loop(CfgB *g, int n, int post) {
  const NodeTable *nt = g->nt;
  int pred = nt_ref(nt, n, "predicate");
  int body = nt_ref(nt, n, "statements");
  int head = cfg_new_block(g), bodyb = cfg_new_block(g), exitb = cfg_new_block(g);
  if (post) cfg_jump_to(g, bodyb); else cfg_jump_to(g, head);
  if (g->nlp < 64) { g->lp_head[g->nlp] = head; g->lp_body[g->nlp] = bodyb; g->lp_exit[g->nlp] = exitb; g->lp_endepth[g->nlp] = g->nen; }
  g->nlp++;
  g->cur = head;
  cfg_lower(g, pred);
  if (g->cur >= 0) { cfg_add_edge(g, g->cur, bodyb); cfg_add_edge(g, g->cur, exitb); }
  g->cur = bodyb;
  cfg_lower_list(g, body);
  cfg_jump_to(g, head);
  g->nlp--;
  g->cur = exitb;
}
static void cfg_lower_begin(CfgB *g, int body, int rescue, int els, int ensure) {
  const NodeTable *nt = g->nt;
  int guarded = rescue >= 0 || ensure >= 0;
  int entry = cfg_fall(g);
  int first = g->nb;              /* blocks made from here on are protected */
  int en = -1;
  if (ensure >= 0) {
    en = cfg_new_block(g);
    if (g->nen < 64) { g->en_block[g->nen] = en; g->en_nt[g->nen] = 0; }
    g->nen++;
  }
  if (guarded) g->protect++;
  cfg_lower_list(g, body);
  int last = g->nb;
  int after_body = g->cur;
  int join = cfg_new_block(g);
  /* an exception in a rescue or else arm reaches the ensure, not the
     rescue arms, so the arms stay protected only while an ensure exists */
  if (ensure < 0 && guarded) g->protect--;
  /* the else arm runs after a body that raised nothing */
  g->cur = after_body;
  if (els >= 0) { cfg_fall(g); cfg_lower(g, els); }
  int normal_exit = g->cur;
  /* rescue arms: any protected block of the body can reach each one, with
     the state at its exit (a write starts a new block, so the state before
     and after every write is one of them) */
  int rexits[64]; int nrex = 0;
  if (g->nbg < 64) { g->bg_entry[g->nbg] = entry; g->bg_endepth[g->nbg] = g->nen; }
  g->nbg++;
  /* An exception goes to the first arm's test. An arm tries its classes
     in order and stops at a match; when none matches, the next arm's
     test runs. So each class can reach its arm, and only the last one's
     miss reaches the next arm. An arm with no class takes StandardError
     and passes anything else on. A miss of the last arm leaves the begin
     (the ensure and the outer rescues see every block made here). */
  int test = -1;
  if (rescue >= 0) {
    test = cfg_new_block(g);
    cfg_add_edge(g, entry, test);
    for (int k = first; k < last; k++) if (k != en) cfg_add_edge(g, k, test);
  }
  for (int r = rescue; r >= 0; r = nt_ref(nt, r, "subsequent")) {
    int rb = cfg_new_block(g);
    int nxt = nt_ref(nt, r, "subsequent") >= 0 ? cfg_new_block(g) : -1;
    g->cur = test;
    { int xn = 0; const int *xs = nt_arr(nt, r, "exceptions", &xn);
      if (xn == 0) { cfg_add_edge(g, g->cur, rb); if (nxt >= 0) cfg_add_edge(g, g->cur, nxt); }
      for (int q = 0; q < xn; q++) {
        if (q > 0) cfg_fall(g);
        cfg_lower(g, xs[q]);
        if (g->cur >= 0) cfg_add_edge(g, g->cur, rb);
      }
      if (xn > 0 && nxt >= 0 && g->cur >= 0) cfg_add_edge(g, g->cur, nxt); }
    test = nxt;
    g->cur = rb;
    int ref = nt_ref(nt, r, "reference");
    if (ref >= 0) {
      if (nt_kind(nt, ref) == NK_LocalVariableTargetNode) cfg_emit(g, EV_WRITE_UNK, ref);
      else cfg_lower(g, ref);
    }
    cfg_lower_list(g, nt_ref(nt, r, "statements"));
    if (nrex < 64) rexits[nrex++] = g->cur;
    else cfg_jump_to(g, join);
  }
  g->nbg--;
  if (ensure >= 0) {
    if (guarded) g->protect--;
    g->nen--;
    /* reached from the normal exit, every rescue exit, and any protected,
       else or rescue block on the exception path */
    if (normal_exit >= 0) cfg_add_edge(g, normal_exit, en);
    for (int k = 0; k < nrex; k++) if (rexits[k] >= 0) cfg_add_edge(g, rexits[k], en);
    cfg_add_edge(g, entry, en);
    for (int k = first; k < g->nb; k++) if (k != en && k != join) cfg_add_edge(g, k, en);
    g->cur = en;
    cfg_lower_list(g, nt_ref(nt, ensure, "statements"));
    int ex = g->cur;
    /* the jumps that left through this ensure go on to their targets */
    int f = g->nen;
    if (f < 64) for (int t = 0; t < g->en_nt[f]; t++) { g->cur = ex; cfg_jump_out(g, g->en_target[f][t], g->en_tdepth[f][t]); }
    g->cur = ex;
    cfg_jump_to(g, join);
  } else {
    if (normal_exit >= 0) cfg_add_edge(g, normal_exit, join);
    for (int k = 0; k < nrex; k++) if (rexits[k] >= 0) cfg_add_edge(g, rexits[k], join);
  }
  g->cur = join;
}

static void cfg_lower(CfgB *g, int n) {
  if (n < 0) return;
  const NodeTable *nt = g->nt;
  NodeKind k = nt_kind(nt, n);
  switch (k) {
  case NK_StatementsNode: cfg_lower_list(g, n); return;
  case NK_ParenthesesNode: cfg_lower(g, nt_ref(nt, n, "body")); return;
  /* own scopes: not entered */
  case NK_DefNode: case NK_ClassNode: case NK_ModuleNode: case NK_SingletonClassNode:
  case NK_BlockNode: case NK_LambdaNode: case NK_DefinedNode:
    return;
  case NK_LocalVariableReadNode:
    if (cfg_owned(g, n)) cfg_emit(g, EV_READ, n);
    return;
  case NK_LocalVariableWriteNode:
    cfg_lower(g, nt_ref(nt, n, "value"));
    if (cfg_owned(g, n)) cfg_emit(g, EV_WRITE, n);
    return;
  case NK_LocalVariableTargetNode:
    if (cfg_owned(g, n)) cfg_emit(g, EV_WRITE_UNK, n);
    return;
  case NK_LocalVariableOperatorWriteNode:
    if (cfg_owned(g, n)) cfg_emit(g, EV_READ, n);
    cfg_lower(g, nt_ref(nt, n, "value"));
    if (cfg_owned(g, n)) cfg_emit(g, EV_OPWRITE, n);
    return;
  case NK_LocalVariableOrWriteNode: case NK_LocalVariableAndWriteNode: {
    /* `x ||= v`: v and the write run only on one side of the test */
    if (cfg_owned(g, n)) cfg_emit(g, EV_READ, n);
    int before = g->cur;
    int join = cfg_new_block(g);
    cfg_fall(g);
    cfg_lower(g, nt_ref(nt, n, "value"));
    if (cfg_owned(g, n)) cfg_emit(g, k == NK_LocalVariableOrWriteNode ? EV_WRITE_UNK : EV_ANDWRITE, n);
    cfg_jump_to(g, join);
    if (before >= 0) cfg_add_edge(g, before, join);
    g->cur = join;
    return;
  }
  case NK_MultiWriteNode: case NK_MultiTargetNode:
    cfg_masgn_operands(g, n, 0);
    cfg_lower(g, nt_ref(nt, n, "value"));
    cfg_masgn_stores(g, n, 0);
    return;
  case NK_InstanceVariableOrWriteNode: case NK_InstanceVariableAndWriteNode:
  case NK_ClassVariableOrWriteNode: case NK_ClassVariableAndWriteNode:
  case NK_GlobalVariableOrWriteNode: case NK_GlobalVariableAndWriteNode:
  case NK_ConstantOrWriteNode: case NK_ConstantAndWriteNode:
  case NK_ConstantPathOrWriteNode: case NK_ConstantPathAndWriteNode:
  case NK_IndexOrWriteNode: case NK_IndexAndWriteNode:
  case NK_CallOrWriteNode: case NK_CallAndWriteNode:
  case NK_CallOperatorWriteNode: case NK_IndexOperatorWriteNode: {
    /* `@a ||= v`, `h[k] &&= v`: the receiver and the arguments run, then
       v runs only on one side of the test. An operator write (`h[k] += v`)
       always runs v, unless it is `a&.b += v` on a nil `a`. */
    cfg_lower(g, nt_ref(nt, n, "target"));
    cfg_lower(g, nt_ref(nt, n, "receiver"));
    int opw = k == NK_CallOperatorWriteNode || k == NK_IndexOperatorWriteNode;
    int sn = nt_safe_nav(nt, n);
    int before = g->cur, join = -1;
    if (sn) { join = cfg_new_block(g); cfg_fall(g); }   /* a nil receiver: nothing after it runs */
    cfg_lower(g, nt_ref(nt, n, "arguments"));
    { int blk = nt_ref(nt, n, "block");
      if (blk >= 0 && nt_kind(nt, blk) != NK_BlockNode) cfg_lower(g, blk); }
    if (!opw) {   /* the test: the value runs on one side of it */
      int tst = g->cur;
      if (join < 0) join = cfg_new_block(g);
      cfg_fall(g);
      if (tst >= 0) cfg_add_edge(g, tst, join);
    }
    cfg_lower(g, nt_ref(nt, n, "value"));
    if (join >= 0) {
      cfg_jump_to(g, join);
      if (sn && before >= 0) cfg_add_edge(g, before, join);
      g->cur = join;
    }
    return;
  }
  case NK_MatchRequiredNode:   /* `v => pat`: binds, or raises */
    cfg_lower(g, nt_ref(nt, n, "value"));
    cfg_lower_pattern_binds(g, nt_ref(nt, n, "pattern"), 0);
    return;
  case NK_MatchPredicateNode:  /* `v in pat` as a value: binds only when it matches */
    cfg_lower(g, nt_ref(nt, n, "value"));
    cfg_lower_arms(g, -1, -1, nt_ref(nt, n, "pattern"));
    return;
  case NK_MatchWriteNode: {    /* named captures: written (nil on no match) */
    cfg_lower(g, nt_ref(nt, n, "call"));
    int tn = 0; const int *t = nt_arr(nt, n, "targets", &tn);
    for (int i = 0; i < tn; i++) cfg_lower(g, t[i]);
    return;
  }
  case NK_IfNode: case NK_UnlessNode: {
    int pred = nt_ref(nt, n, "predicate");
    cfg_lower(g, pred);
    int binds = -1;
    if (pred >= 0 && nt_kind(nt, pred) == NK_MatchPredicateNode) binds = nt_ref(nt, pred, "pattern");
    int then = nt_ref(nt, n, "statements");
    int els = nt_ref(nt, n, k == NK_IfNode ? "subsequent" : "else_clause");
    if (k == NK_UnlessNode) cfg_lower_arms(g, els >= 0 ? nt_ref(nt, els, "statements") : -1, then, -1);
    else cfg_lower_arms(g, then, els, binds);
    return;
  }
  case NK_ElseNode: cfg_lower_list(g, nt_ref(nt, n, "statements")); return;
  case NK_AndNode: case NK_OrNode: {
    cfg_lower(g, nt_ref(nt, n, "left"));
    int before = g->cur;
    int join = cfg_new_block(g);
    cfg_fall(g);
    cfg_lower(g, nt_ref(nt, n, "right"));
    cfg_jump_to(g, join);
    if (before >= 0) cfg_add_edge(g, before, join);
    g->cur = join;
    return;
  }
  case NK_CaseNode: case NK_CaseMatchNode: {
    cfg_lower(g, nt_ref(nt, n, "predicate"));
    int join = cfg_new_block(g);
    int test = cfg_fall(g);                         /* the first arm's test */
    int wn = 0; const int *w = nt_arr(nt, n, "conditions", &wn);
    for (int i = 0; i < wn; i++) {
      int next = cfg_new_block(g);                  /* no match: the next test */
      int arm = cfg_new_block(g);
      g->cur = test;
      if (k == NK_CaseNode) {
        /* `when a, b`: the conditions run in order and the first match
           runs the arm, so each condition can reach it; only the last
           one's miss goes on to the next arm's test */
        int cn = 0; const int *cv = nt_arr(nt, w[i], "conditions", &cn);
        for (int q = 0; q < cn; q++) {
          if (q > 0) cfg_fall(g);
          cfg_lower(g, cv[q]);
          if (g->cur >= 0) cfg_add_edge(g, g->cur, arm);
        }
        if (g->cur >= 0) cfg_add_edge(g, g->cur, next);
        g->cur = arm;
      } else {
        int pat = nt_ref(nt, w[i], "pattern");
        int guard = -1;
        if (pat >= 0 && (nt_kind(nt, pat) == NK_IfNode || nt_kind(nt, pat) == NK_UnlessNode)) {
          guard = nt_ref(nt, pat, "predicate"); pat = nt_ref(nt, pat, "statements");
        }
        cfg_add_edge(g, test, arm); cfg_add_edge(g, test, next);
        g->cur = arm;
        cfg_lower_pattern_binds(g, pat, 0);
        if (guard >= 0) {
          cfg_lower(g, guard);
          int body = cfg_new_block(g);
          cfg_add_edge(g, g->cur, body); cfg_add_edge(g, g->cur, next);
          g->cur = body;
        }
      }
      cfg_lower_list(g, nt_ref(nt, w[i], "statements"));
      cfg_jump_to(g, join);
      test = next;
    }
    g->cur = test;
    int els = nt_ref(nt, n, "else_clause");
    if (els >= 0) { cfg_lower(g, els); cfg_jump_to(g, join); }
    else if (k == NK_CaseNode) cfg_jump_to(g, join);   /* no arm: nil */
    else cfg_jump_to(g, g->exit_blk);                  /* no arm: NoMatchingPatternError */
    g->cur = join;
    return;
  }
  case NK_WhileNode: case NK_UntilNode:
    cfg_lower_loop(g, n, nt_loop_post(nt, n));
    return;
  case NK_ForNode: {
    cfg_lower(g, nt_ref(nt, n, "collection"));
    int head = cfg_new_block(g), bodyb = cfg_new_block(g), exitb = cfg_new_block(g);
    cfg_jump_to(g, head);
    cfg_add_edge(g, head, bodyb); cfg_add_edge(g, head, exitb);
    if (g->nlp < 64) { g->lp_head[g->nlp] = head; g->lp_body[g->nlp] = bodyb; g->lp_exit[g->nlp] = exitb; g->lp_endepth[g->nlp] = g->nen; }
    g->nlp++;
    g->cur = bodyb;
    cfg_lower(g, nt_ref(nt, n, "index"));
    /* `redo` runs the body again on the same element: it goes to the
       block after the index write, not the one that makes it */
    { int again = cfg_fall(g);
      if (g->nlp <= 64) g->lp_body[g->nlp - 1] = again; }
    cfg_lower_list(g, nt_ref(nt, n, "statements"));
    cfg_jump_to(g, head);
    g->nlp--;
    g->cur = exitb;
    return;
  }
  case NK_BeginNode:
    cfg_lower_begin(g, nt_ref(nt, n, "statements"), nt_ref(nt, n, "rescue_clause"),
                nt_ref(nt, n, "else_clause"), nt_ref(nt, n, "ensure_clause"));
    return;
  case NK_RescueModifierNode: {
    /* `a rescue b`: b runs when a raised */
    int entry = cfg_fall(g);
    int first = g->nb;
    g->protect++;
    cfg_lower(g, nt_ref(nt, n, "expression"));
    g->protect--;
    int last = g->nb;
    int join = cfg_new_block(g);
    cfg_jump_to(g, join);
    int rb = cfg_new_block(g);
    cfg_add_edge(g, entry, rb);
    for (int q = first; q < last; q++) cfg_add_edge(g, q, rb);
    g->cur = rb;
    cfg_lower(g, nt_ref(nt, n, "rescue_expression"));
    cfg_jump_to(g, join);
    g->cur = join;
    return;
  }
  case NK_ReturnNode:
    cfg_lower(g, nt_ref(nt, n, "arguments"));
    cfg_jump_out(g, g->exit_blk, 0);
    return;
  case NK_BreakNode: case NK_NextNode: case NK_RedoNode: {
    if (k != NK_RedoNode) cfg_lower(g, nt_ref(nt, n, "arguments"));
    int in_loop = g->nlp > 0 && g->nlp <= 64;
    int t = !in_loop ? g->exit_blk : k == NK_BreakNode ? g->lp_exit[g->nlp - 1]
          : k == NK_NextNode ? g->lp_head[g->nlp - 1] : g->lp_body[g->nlp - 1];
    cfg_jump_out(g, t, in_loop ? g->lp_endepth[g->nlp - 1] : 0);
    return;
  }
  case NK_RetryNode:
    if (g->nbg > 0 && g->nbg <= 64) cfg_jump_out(g, g->bg_entry[g->nbg - 1], g->bg_endepth[g->nbg - 1]);
    else cfg_jump_out(g, g->exit_blk, 0);
    return;
  case NK_CallNode: {
    cfg_lower(g, nt_ref(nt, n, "receiver"));
    /* `m(&x)` reads x after the arguments; a block is its own scope.
       `a&.m(x = v, &e)`: the arguments and `e` run only when a is not nil */
    int before = -1, join = -1;
    if (nt_safe_nav(nt, n)) { before = g->cur; join = cfg_new_block(g); cfg_fall(g); }
    cfg_lower(g, nt_ref(nt, n, "arguments"));
    { int blk = nt_ref(nt, n, "block");
      if (blk >= 0 && nt_kind(nt, blk) != NK_BlockNode) cfg_lower(g, blk); }
    if (join >= 0) {
      cfg_jump_to(g, join);
      if (before >= 0) cfg_add_edge(g, before, join);
      g->cur = join;
    }
    /* the block is its own scope; a bare raise/exit that reaches Kernel's
       never returns (a method of the program with the name does) */
    if (!g->ssa && an_call_diverges(g->c, n)) cfg_jump_out(g, g->exit_blk, 0);
    return;
  }
  default:
    cfg_lower_children(g, n);
    return;
  }
}

/* ---- the closures' writes: a local they write keeps no fact ---- */
static void cfg_mark_captured(CfgB *g, int n, int in_closure, int depth) {
  if (n < 0 || depth > 200) return;
  const NodeTable *nt = g->nt;
  NodeKind k = nt_kind(nt, n);
  if (k == NK_DefNode || k == NK_ClassNode || k == NK_ModuleNode || k == NK_SingletonClassNode) return;
  if (k == NK_BlockNode || k == NK_LambdaNode) in_closure = 1;
  if (in_closure && comp_is_local_write(k)) {
    int li = cfg_local_index(g, nt_str(nt, n, "name"));
    if (li >= 0) g->captured[li] = 1;
  }
  for (int i = 0; i < nt_num_refs(nt, n); i++) cfg_mark_captured(g, nt_ref_at(nt, n, i), in_closure, depth + 1);
  for (int i = 0; i < nt_num_arrs(nt, n); i++) {
    int an = 0; const int *av = nt_arr_at(nt, n, i, &an);
    for (int q = 0; q < an; q++) cfg_mark_captured(g, av[q], in_closure, depth + 1);
  }
}

/* ---- the dataflow ---- */
/* the value is laid out as a plain Integer or Float (no nil beside it, no
   box): the representation layer's answer, which codegen also follows */
static int cfg_narrowable(Compiler *c, int value, TyKind t) {
  if (t != TY_INT && t != TY_FLOAT) return 0;
  if (nullable_int_value(c, value)) return 0;
  Repr r = repr_of(c, value);
  return r.kind == RK_SCALAR && r.as_ty == t;
}
static unsigned char cfg_write_kind(CfgB *g, int kind, int node) {
  if (!g->want_types) return TY_BOT;
  Compiler *c = g->c;
  if (kind == EV_WRITE) {
    int v = nt_ref(g->nt, node, "value");
    TyKind t = v >= 0 ? infer_type(c, v) : TY_UNKNOWN;
    return cfg_narrowable(c, v, t) ? (unsigned char)(t + 1) : TY_BOT;
  }
  if (kind == EV_OPWRITE) {
    TyKind t = infer_type(c, node);
    return cfg_narrowable(c, node, t) ? (unsigned char)(t + 1) : TY_BOT;
  }
  return TY_BOT;
}
static void cfg_transfer(CfgB *g, int bi, unsigned char *ty, unsigned char *unset, int record) {
  CBlock *bl = &g->b[bi];
  for (int e = 0; e < bl->nev; e++) {
    int kind = bl->ev_kind[e], node = bl->ev_node[e];
    int li = cfg_local_index(g, nt_str(g->nt, node, "name"));
    if (li < 0) continue;
    if (kind == EV_READ) {
      if (record && !g->captured[li]) {
        if (node < fc_cap) {
          fc_unset[node] = unset[li] ? 1 : 0;
          fc_ty[node] = (ty[li] != TY_TOP && ty[li] != TY_BOT) ? (unsigned char)(ty[li] - 1) : TY_UNKNOWN;
          st_reads++;
        }
      }
      continue;
    }
    if (kind == EV_ANDWRITE) { ty[li] = TY_BOT; continue; }   /* unset stays */
    unset[li] = 0;
    ty[li] = cfg_write_kind(g, kind, node);
  }
}
static void cfg_meet_into(unsigned char *ty, unsigned char *unset, const unsigned char *oty, const unsigned char *ounset, int n, int *changed) {
  for (int i = 0; i < n; i++) {
    unsigned char t = ty[i], o = oty[i], r;
    if (t == TY_TOP) r = o; else if (o == TY_TOP || o == t) r = t; else r = TY_BOT;
    if (r != t) { ty[i] = r; *changed = 1; }
    if (ounset[i] && !unset[i]) { unset[i] = 1; *changed = 1; }
  }
}

static void cfg_solve(CfgB *g) {
  int nl = g->s->nlocals, nb = g->nb;
  size_t sz = (size_t)nl;
  unsigned char *in_ty = calloc((size_t)nb * sz + 1, 1), *in_un = calloc((size_t)nb * sz + 1, 1);
  unsigned char *out_ty = calloc((size_t)nb * sz + 1, 1), *out_un = calloc((size_t)nb * sz + 1, 1);
  char *inq = calloc((size_t)nb + 1, 1);
  int *queue = malloc(sizeof(int) * ((size_t)nb * 4 + 4));
  if (!in_ty || !in_un || !out_ty || !out_un || !inq || !queue) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  /* entry: every local is unset and nil; a parameter holds any kind */
  for (int i = 0; i < nl; i++) {
    LocalVar *lv = &g->s->locals[i];
    if (lv->is_param) { in_ty[i] = TY_BOT; in_un[i] = 0; }
    else { in_ty[i] = (unsigned char)(TY_NIL + 1); in_un[i] = 1; }
  }
  g->b[0].reached = 1;
  int qh = 0, qt = 0;
  queue[qt++] = 0; inq[0] = 1;
  int steps = 0;
  while (qh < qt) {
    int bi = queue[qh++]; inq[bi] = 0;
    if (++steps > nb * 64 + 1000) break;   /* a backstop: the meet only lowers, so this does not fire */
    memcpy(out_ty + (size_t)bi * sz, in_ty + (size_t)bi * sz, sz);
    memcpy(out_un + (size_t)bi * sz, in_un + (size_t)bi * sz, sz);
    cfg_transfer(g, bi, out_ty + (size_t)bi * sz, out_un + (size_t)bi * sz, 0);
    CBlock *bl = &g->b[bi];
    for (int k = 0; k < bl->nsucc; k++) {
      int sb = bl->succ[k];
      int changed = !g->b[sb].reached;
      g->b[sb].reached = 1;
      cfg_meet_into(in_ty + (size_t)sb * sz, in_un + (size_t)sb * sz, out_ty + (size_t)bi * sz, out_un + (size_t)bi * sz, nl, &changed);
      if (changed && !inq[sb]) {
        if (qt >= nb * 4 + 2) {   /* compact the queue */
          int w = 0; for (int r = qh; r < qt; r++) queue[w++] = queue[r];
          qh = 0; qt = w;
        }
        queue[qt++] = sb; inq[sb] = 1;
      }
    }
  }
  /* the final pass records the facts each read sees */
  unsigned char *ty = malloc(sz + 1), *un = malloc(sz + 1);
  if (!ty || !un) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  for (int bi = 0; bi < nb; bi++) {
    if (!g->b[bi].reached) continue;
    memcpy(ty, in_ty + (size_t)bi * sz, sz);
    memcpy(un, in_un + (size_t)bi * sz, sz);
    cfg_transfer(g, bi, ty, un, 1);
  }
  free(ty); free(un);
  free(in_ty); free(in_un); free(out_ty); free(out_un); free(inq); free(queue);
}

static void cfg_build_scope(Compiler *c, Scope *s, int si, int want_types) {
  if (s->body < 0 || s->cs_synth) return;
  CfgB g; memset(&g, 0, sizeof g);
  g.c = c; g.nt = c->nt; g.s = s; g.si = si; g.want_types = want_types;
  g.captured = calloc((size_t)s->nlocals + 1, 1);
  if (!g.captured) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  for (int i = 0; i < s->nlocals; i++)
    if (s->locals[i].is_cell || s->locals[i].proc_rebinds) g.captured[i] = 1;
  cfg_mark_captured(&g, s->body, 0, 0);
  g.cur = cfg_new_block(&g);          /* block 0: entry */
  g.exit_blk = cfg_new_block(&g);
  cfg_lower(&g, s->body);
  cfg_jump_to(&g, g.exit_blk);
  st_blocks += g.nb; st_methods++;
  cfg_solve(&g);
  for (int i = 0; i < g.nb; i++) { free(g.b[i].ev_kind); free(g.b[i].ev_node); free(g.b[i].succ); free(g.b[i].ev_def); }
  free(g.b); free(g.captured);
}

static void cfg_facts_build(Compiler *c, int want_types) {
  const NodeTable *nt = c->nt;
  free(fc_unset); free(fc_ty);
  fc_cap = nt->count;
  fc_unset = malloc((size_t)fc_cap + 1); fc_ty = calloc((size_t)fc_cap + 1, 1);
  if (!fc_unset || !fc_ty) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  memset(fc_unset, 2, (size_t)fc_cap + 1);
  fc_c = c; fc_version = nt->version;
  st_blocks = st_methods = st_reads = 0;
  for (int si = 0; si < c->nscopes; si++) {
    Scope *s = &c->scopes[si];
    /* method bodies and the top level: a block's locals live in its owner */
    if (si != 0 && s->def_node < 0) continue;
    if (si != 0 && nt_kind(nt, s->def_node) != NK_DefNode) continue;
    cfg_build_scope(c, s, si, want_types);
  }
  if (cfg_log()) fprintf(stderr, "spinel-cfg: methods=%d blocks=%d reads=%d types=%d\n", st_methods, st_blocks, st_reads, want_types);
}

void cfg_facts_ensure(Compiler *c) {
  if (cfg_disabled()) return;
  if (fc_c == c && fc_version == c->nt->version && fc_unset) return;
  cfg_facts_build(c, 0);
}
/* -1 no fact, else whether the read may run before any write */
int cfg_read_unset(int node) {
  if (cfg_disabled() || !fc_unset || node < 0 || node >= fc_cap) return -1;
  if (fc_unset[node] == 2) return -1;
  return fc_unset[node];
}
/* a read that is a direct element of an array or hash literal keeps the
   box (isa_mark_reads' rule: a typed element retypes the literal) */
static int direct_literal_elem(Compiler *c, const int *par, int node) {
  if (!par) return 0;
  int p = par[node];
  if (p < 0) return 0;
  NodeKind pk = nt_kind(c->nt, p);
  if (pk == NK_ArrayNode || pk == NK_HashNode) return 1;
  if (pk == NK_AssocNode) { int pp = par[p]; return pp >= 0 && nt_kind(c->nt, pp) == NK_HashNode; }
  return 0;
}
/* does the subtree under n write the local nm */
static int cfg_subtree_writes(const NodeTable *nt, int n, const char *nm) {
  if (n < 0) return 0;
  NodeKind k = nt_kind(nt, n);
  if (k == NK_LocalVariableWriteNode || k == NK_LocalVariableTargetNode ||
      k == NK_LocalVariableOperatorWriteNode || k == NK_LocalVariableAndWriteNode ||
      k == NK_LocalVariableOrWriteNode) {
    const char *wn = nt_str(nt, n, "name");
    if (wn && sp_streq(wn, nm)) return 1;
  }
  int nr = nt_num_refs(nt, n);
  for (int i = 0; i < nr; i++) if (cfg_subtree_writes(nt, nt_ref_at(nt, n, i), nm)) return 1;
  int na = nt_num_arrs(nt, n);
  for (int i = 0; i < na; i++) {
    int cnt = 0; const int *ids = nt_arr_at(nt, n, i, &cnt);
    for (int j = 0; j < cnt; j++) if (cfg_subtree_writes(nt, ids[j], nm)) return 1;
  }
  return 0;
}
/* A read whose own statement also writes the local (`(i = 1; a)[i] = 9`):
   the C that codegen prints for the statement orders the two as C
   arguments, which an unboxed read would reorder. Left boxed. */
static int cfg_statement_writes(Compiler *c, const int *par, int r, const char *nm) {
  if (!par) return 0;
  int n = r;
  while (par[n] >= 0 && nt_kind(c->nt, par[n]) != NK_StatementsNode) n = par[n];
  if (n == r) return 0;
  /* `x = x + 2`: the statement's own store lands after its value is
     computed, so only the value is looked at */
  NodeKind k = nt_kind(c->nt, n);
  if (k == NK_LocalVariableWriteNode || k == NK_LocalVariableOperatorWriteNode ||
      k == NK_LocalVariableAndWriteNode || k == NK_LocalVariableOrWriteNode) {
    const char *wn = nt_str(c->nt, n, "name");
    if (wn && sp_streq(wn, nm)) n = nt_ref(c->nt, n, "value");
  }
  return cfg_subtree_writes(c->nt, n, nm);
}
int cfg_narrow_reads(Compiler *c) {
  if (cfg_disabled()) return 0;
  cfg_facts_build(c, 1);
  st_narrowed = 0;
  int *par = an_parent_map(c->nt);
  NT_FOREACH_KIND(c->nt, NK_LocalVariableReadNode, r) {
    if (r >= fc_cap || fc_ty[r] == TY_UNKNOWN) continue;
    const char *nm = nt_str(c->nt, r, "name");
    LocalVar *lv = nm ? scope_local(comp_scope_of(c, r), nm) : NULL;
    if (!lv || lv->type != TY_POLY || lv->is_cell || lv->proc_rebinds) continue;
    if (c->nilnarrow[r] != TY_UNKNOWN) continue;
    if (direct_literal_elem(c, par, r)) continue;
    /* the prelude's methods convert their arguments through #to_int and
       friends at sites the reachability pass finds by their boxed reads:
       their reads stay boxed */
    if (nt_int(c->nt, r, "node_bi", 0) != 0) continue;
    if (cfg_statement_writes(c, par, r, nm)) continue;
    comp_set_nilnarrow(c, r, (TyKind)fc_ty[r]);
    st_narrowed++;
  }
  free(par);
  if (cfg_log()) fprintf(stderr, "spinel-cfg: narrowed=%d\n", st_narrowed);
  return st_narrowed;
}

/* ---- SSA: one name per write, and the names that must share a slot ----

   Static single assignment form gives every write of a local its own name
   (x1, x2, ...) and puts a phi where two names meet at a join:

     x1 = 0
     while ...                 x2 = phi(x1, x3)   (the loop head)
       x3 = x2 + i
     y1 = x2 * 2
     x4 = "sum=#{y1}"

   Spinel types a local once for its whole life, so `x` above is poly. In
   SSA each name has one type: x1, x2, x3 are Integers and x4 is a String.

   Out of SSA, the names a phi joins must share one C slot (the phi is a
   copy that is free only when both sides are the same slot). The groups
   this leaves are the local's webs: here {x1, x2, x3} and {x4}. A web is
   the set of writes some read can see, closed over shared reads, which is
   what this pass computes, with reaching definitions on the CFG above:
   every read joins the writes that reach it into one web.

   Each web after the first gets a new name in the tree (`x__ssa1`), before
   the scope's locals are registered, so the rest of Spinel sees two locals
   and types each one on its own. Left alone: a local a closure, a pattern,
   a `defined?` or a regexp capture refers to (the graph does not model
   those reads), and every local of a method that reads its frame by name
   (binding, eval, local_variables, a bare `super`). */

static int ssa_disabled(void) {
  static int v = -1;
  if (v < 0) v = getenv("SP_NO_SSA") != NULL || cfg_disabled();
  return v;
}
static int ssa_log(void) {
  static int v = -1;
  if (v < 0) v = getenv("SP_SSA_LOG") != NULL;
  return v;
}

typedef struct {
  const char **nm; int n, cap;
  char *bad;          /* left alone: the graph cannot see every use */
  int method_bad;
  int *nodes, *node_name, nnodes, cnodes;   /* every local node of the body */
  int *pw, *pr, np, cp;   /* a write and a read of the same local in its value */
} SsaNames;

static int ssa_name_add(SsaNames *t, const char *nm) {
  for (int i = 0; i < t->n; i++) if (sp_streq(t->nm[i], nm)) return i;
  if (t->n >= t->cap) {
    t->cap = t->cap ? t->cap * 2 : 16;
    t->nm = realloc(t->nm, sizeof(char *) * (size_t)t->cap);
    t->bad = realloc(t->bad, (size_t)t->cap);
    if (!t->nm || !t->bad) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  }
  /* a copy: renaming a node frees the string its name was read from */
  t->nm[t->n] = strdup(nm); t->bad[t->n] = 0;
  if (!t->nm[t->n]) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  return t->n++;
}
static int ssa_local_node(NodeKind k) {
  return k == NK_LocalVariableReadNode || k == NK_LocalVariableWriteNode ||
         k == NK_LocalVariableTargetNode || k == NK_LocalVariableOperatorWriteNode ||
         k == NK_LocalVariableOrWriteNode || k == NK_LocalVariableAndWriteNode;
}
/* a name that reads the method's frame by name */
static int ssa_frame_name(const char *nm) {
  static const char *const names[] = { "binding", "local_variables", "eval", "instance_eval",
    "class_eval", "module_eval", "local_variable_get", "local_variable_set",
    "local_variable_defined?", NULL };
  if (!nm) return 0;
  for (int i = 0; names[i]; i++) if (sp_streq(nm, names[i])) return 1;
  return 0;
}
/* the subtree of `n` writes local `nm`, outside `skip` (a nested def or
   class has locals of its own; a tree too deep to walk counts as a write) */
static int ssa_writes(const NodeTable *nt, int n, const char *nm, int skip, int depth) {
  if (n < 0 || n == skip) return 0;
  if (depth > 2000) return 1;
  NodeKind k = nt_kind(nt, n);
  if (k == NK_DefNode || k == NK_ClassNode || k == NK_ModuleNode || k == NK_SingletonClassNode) return 0;
  if (k != NK_LocalVariableReadNode && ssa_local_node(k) && sp_streq(nt_str(nt, n, "name"), nm)) return 1;
  for (int i = 0; i < nt_num_refs(nt, n); i++)
    if (ssa_writes(nt, nt_ref_at(nt, n, i), nm, skip, depth + 1)) return 1;
  for (int i = 0; i < nt_num_arrs(nt, n); i++) {
    int an = 0; const int *av = nt_arr_at(nt, n, i, &an);
    for (int q = 0; q < an; q++) if (ssa_writes(nt, av[q], nm, skip, depth + 1)) return 1;
  }
  return 0;
}
/* A call on a local reads the receiver first and holds it while the
   arguments run. A store into a typed Array that needs a wider one (an
   Array of Integers given a String) makes the wider copy and writes it
   back to the slot of the local it was called on. When an argument writes
   the same local (`h[i] = (h = h; v)`), the write and the receiver are in
   two webs: the copy goes back to the receiver's slot, and the reads after
   the call see the other slot, which holds the old Array. So that local
   keeps one slot, as on master. `root` is the subtree that runs while the
   receiver is held: the call itself, or a multiple assignment whose target
   is the call (`x[0], x = v, w` runs both receivers before any store). */
static void ssa_recv_held(Compiler *c, SsaNames *t, int root, int n) {
  const NodeTable *nt = c->nt;
  int rcv = nt_ref(nt, n, "receiver");
  if (rcv < 0) return;
  int rv = nt_paren_single(nt, rcv);
  if (rv < 0 || nt_kind(nt, rv) != NK_LocalVariableReadNode) return;
  const char *nm = nt_str(nt, rv, "name");
  if (nm && ssa_writes(nt, root, nm, rcv, 0)) { int i = ssa_name_add(t, nm); t->bad[i] = 1; }
}
/* the targets of a multiple assignment that are calls (`x[0]`, `x.a`) */
static void ssa_masgn_targets(Compiler *c, SsaNames *t, int root, int n, int depth) {
  const NodeTable *nt = c->nt;
  if (n < 0 || depth > 200) return;
  NodeKind k = nt_kind(nt, n);
  if (k == NK_IndexTargetNode || k == NK_CallTargetNode) { ssa_recv_held(c, t, root, n); return; }
  if (k == NK_SplatNode) { ssa_masgn_targets(c, t, root, nt_ref(nt, n, "expression"), depth + 1); return; }
  if (k != NK_MultiWriteNode && k != NK_MultiTargetNode) return;
  static const char *const lists[] = { "lefts", "rights" };
  for (int i = 0; i < 2; i++) {
    int an = 0; const int *av = nt_arr(nt, n, lists[i], &an);
    for (int q = 0; q < an; q++) ssa_masgn_targets(c, t, root, av[q], depth + 1);
  }
  ssa_masgn_targets(c, t, root, nt_ref(nt, n, "rest"), depth + 1);
}
/* The reads of local `nm` inside `v`, the value write `w` stores. Such a
   write can store the object the local already held: `x = x.push(3)`,
   `x = x.itself`, `x = x.to_s` for a String. Spinel follows a value
   through the writes of a local (to widen an Array where it is built,
   before a store of another kind reaches it), not through what a method
   returns. So with the write in a web of its own, a store through the
   new slot does not reach the Array the old slot got from the caller. The
   write stays in the web of the writes those reads see (ssa_method). */
static void ssa_value_reads(const NodeTable *nt, SsaNames *t, int w, int v, const char *nm, int depth) {
  if (v < 0) return;
  if (depth > 2000) { t->method_bad = 1; return; }
  NodeKind k = nt_kind(nt, v);
  if (k == NK_DefNode || k == NK_ClassNode || k == NK_ModuleNode || k == NK_SingletonClassNode) return;
  if (k == NK_LocalVariableReadNode && sp_streq(nt_str(nt, v, "name"), nm)) {
    if (t->np >= t->cp) {
      t->cp = t->cp ? t->cp * 2 : 16;
      t->pw = realloc(t->pw, sizeof(int) * (size_t)t->cp);
      t->pr = realloc(t->pr, sizeof(int) * (size_t)t->cp);
      if (!t->pw || !t->pr) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
    }
    t->pw[t->np] = w; t->pr[t->np] = v; t->np++;
    return;
  }
  for (int i = 0; i < nt_num_refs(nt, v); i++) ssa_value_reads(nt, t, w, nt_ref_at(nt, v, i), nm, depth + 1);
  for (int i = 0; i < nt_num_arrs(nt, v); i++) {
    int an = 0; const int *av = nt_arr_at(nt, v, i, &an);
    for (int q = 0; q < an; q++) ssa_value_reads(nt, t, w, av[q], nm, depth + 1);
  }
}
/* the plain local targets of a multiple assignment or a `for`, each with
   the value it is given */
static void ssa_target_reads(const NodeTable *nt, SsaNames *t, int n, int v, int depth) {
  if (n < 0 || depth > 200) return;
  NodeKind k = nt_kind(nt, n);
  if (k == NK_LocalVariableTargetNode) {
    const char *nm = nt_str(nt, n, "name");
    if (nm) ssa_value_reads(nt, t, n, v, nm, 0);
    return;
  }
  if (k == NK_SplatNode) { ssa_target_reads(nt, t, nt_ref(nt, n, "expression"), v, depth + 1); return; }
  if (k != NK_MultiWriteNode && k != NK_MultiTargetNode) return;
  static const char *const lists[] = { "lefts", "rights" };
  for (int i = 0; i < 2; i++) {
    int an = 0; const int *av = nt_arr(nt, n, lists[i], &an);
    for (int q = 0; q < an; q++) ssa_target_reads(nt, t, av[q], v, depth + 1);
  }
  ssa_target_reads(nt, t, nt_ref(nt, n, "rest"), v, depth + 1);
}
/* every local node of the body (not a nested def or class); `opaque` is a
   subtree whose uses the graph does not model */
static void ssa_collect(Compiler *c, Scope *s, int n, SsaNames *t, int in_closure, int opaque, int depth) {
  if (n < 0) return;
  if (depth > 2000) { t->method_bad = 1; return; }
  const NodeTable *nt = c->nt;
  NodeKind k = nt_kind(nt, n);
  if (k == NK_DefNode || k == NK_ClassNode || k == NK_ModuleNode || k == NK_SingletonClassNode) return;
  if (k == NK_BlockNode || k == NK_LambdaNode) in_closure = 1;
  if (k == NK_DefinedNode || k == NK_MatchWriteNode) opaque = 1;
  if (k == NK_ForwardingSuperNode) t->method_bad = 1;   /* passes the parameters by name */
  if (k == NK_CallNode && ssa_frame_name(nt_str(nt, n, "name"))) t->method_bad = 1;
  if (k == NK_SymbolNode && ssa_frame_name(nt_str(nt, n, "value"))) t->method_bad = 1;
  if (k == NK_StringNode && ssa_frame_name(nt_str(nt, n, "content"))) t->method_bad = 1;   /* send("binding") */
  ssa_recv_held(c, t, n, n);
  if (k == NK_MultiWriteNode) ssa_masgn_targets(c, t, n, n, 0);
  if (k == NK_LocalVariableWriteNode && nt_str(nt, n, "name"))
    ssa_value_reads(nt, t, n, nt_ref(nt, n, "value"), nt_str(nt, n, "name"), 0);
  if (k == NK_MultiWriteNode) ssa_target_reads(nt, t, n, nt_ref(nt, n, "value"), 0);
  if (k == NK_ForNode) ssa_target_reads(nt, t, nt_ref(nt, n, "index"), nt_ref(nt, n, "collection"), 0);
  if (ssa_local_node(k)) {
    const char *nm = nt_str(nt, n, "name");
    if (nm) {
      int i = ssa_name_add(t, nm);
      if (in_closure || opaque || strstr(nm, "__") || comp_scope_of(c, n) != s) t->bad[i] = 1;
      if (t->nnodes >= t->cnodes) {
        t->cnodes = t->cnodes ? t->cnodes * 2 : 32;
        t->nodes = realloc(t->nodes, sizeof(int) * (size_t)t->cnodes);
        t->node_name = realloc(t->node_name, sizeof(int) * (size_t)t->cnodes);
        if (!t->nodes || !t->node_name) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
      }
      t->nodes[t->nnodes] = n; t->node_name[t->nnodes] = i; t->nnodes++;
    }
  }
  /* a pattern binds and reads (`^x`) in ways the graph keeps no events for */
  int pat = (k == NK_InNode || k == NK_MatchRequiredNode || k == NK_MatchPredicateNode)
            ? nt_ref(nt, n, "pattern") : -1;
  for (int i = 0; i < nt_num_refs(nt, n); i++) {
    int ch = nt_ref_at(nt, n, i);
    ssa_collect(c, s, ch, t, in_closure, opaque || (ch >= 0 && ch == pat), depth + 1);
  }
  for (int i = 0; i < nt_num_arrs(nt, n); i++) {
    int an = 0; const int *av = nt_arr_at(nt, n, i, &an);
    for (int q = 0; q < an; q++) ssa_collect(c, s, av[q], t, in_closure, opaque, depth + 1);
  }
}

/* union-find over definitions */
static int ssa_find(int *uf, int d) {
  while (uf[d] != d) { uf[d] = uf[uf[d]]; d = uf[d]; }
  return d;
}
static void ssa_union(int *uf, int a, int b) {
  a = ssa_find(uf, a); b = ssa_find(uf, b);
  if (a != b) { if (a < b) uf[b] = a; else uf[a] = b; }
}

static int *g_ssa_rd;      /* per node: a definition its read sees, or -1 */
static int *g_ssa_wd;      /* per node: the definition its write makes, or -1 */
static int g_ssa_rd_cap;
static char *g_ssa_seen;   /* per node: the graph keeps an event for it */
static int st_ssa_methods, st_ssa_split, st_ssa_renamed, st_ssa_phis;

/* Each split made, so ssa_check_splits can read the types its slots got.
   A split is kept only when every slot of it is a plain value (Integer,
   Float, String, Symbol, true/false, nil). An Array or Hash slot is where
   Spinel's typed containers can refuse an element a poly slot took, so a
   split that types one is undone: the compile runs again with that local
   left whole (SP_SSA_VETO). */
typedef struct { int scope; char *key; char **names; int nnames; } SsaSplit;
static SsaSplit *g_ssa_splits;
static int g_nssa_splits, g_cssa_splits;
char **g_sp_argv;   /* main's argv, to run the compile again */

/* a split's name across runs: the def's line, the method and the local */
static void ssa_split_key(Compiler *c, Scope *s, const char *nm, char *out, size_t cap) {
  snprintf(out, cap, "%d:%s:%s", (int)nt_int(c->nt, s->def_node, "node_line", 0), s->name ? s->name : "?", nm);
}
static int ssa_vetoed(const char *key) {
  const char *v = getenv("SP_SSA_VETO");
  if (!v || !*v) return 0;
  size_t kl = strlen(key);
  for (const char *p = v; *p; ) {
    const char *e = strchr(p, ',');
    size_t l = e ? (size_t)(e - p) : strlen(p);
    if (l == kl && strncmp(p, key, kl) == 0) return 1;
    if (!e) break;
    p = e + 1;
  }
  return 0;
}

static int ssa_line(Compiler *c, int node) {
  return node >= 0 ? (int)nt_int(c->nt, node, "node_line", 0) : 0;
}
/* a read: it joins every definition of its name that reaches it */
static void ssa_read(CfgB *g, int node, int li, const uint64_t *cur, const int *dl,
                     const int *dl_off, int *uf, const int *def_node) {
  int rep = -1, n = 0;
  for (int q = dl_off[li]; q < dl_off[li + 1]; q++) {
    int dd = dl[q];
    if (!((cur[dd / 64] >> (dd % 64)) & 1)) continue;
    if (rep < 0) rep = dd; else ssa_union(uf, rep, dd);
    n++;
  }
  if (node < g_ssa_rd_cap) g_ssa_rd[node] = rep;
  if (n >= 2) {
    st_ssa_phis++;
    if (ssa_log()) {
      fprintf(stderr, "spinel-ssa:   line %d reads `%s` through a phi of", ssa_line(g->c, node), g->sn[li]);
      for (int q = dl_off[li]; q < dl_off[li + 1]; q++) {
        int dd = dl[q];
        if (!((cur[dd / 64] >> (dd % 64)) & 1)) continue;
        if (def_node[dd] < 0) fprintf(stderr, " entry");
        else fprintf(stderr, " line-%d", ssa_line(g->c, def_node[dd]));
      }
      fprintf(stderr, "\n");
    }
  }
}

static void ssa_method(Compiler *c, Scope *s) {
  const NodeTable *nt = c->nt;
  int body = s->body >= 0 ? s->body : nt_ref(nt, s->def_node, "body");
  if (body < 0) return;
  SsaNames t; memset(&t, 0, sizeof t);
  ssa_collect(c, s, body, &t, 0, 0, 0);
  if (t.method_bad || t.n == 0) {
    for (int i = 0; i < t.n; i++) free((char *)t.nm[i]);
    free(t.nm); free(t.bad); free(t.nodes); free(t.node_name); free(t.pw); free(t.pr);
    return;
  }
  /* a parameter's entry definition is the argument, any other's is nil:
     either way the web it is in keeps the name */
  CfgB g; memset(&g, 0, sizeof g);
  g.c = c; g.nt = nt; g.s = s; g.si = s->idx; g.ssa = 1;
  g.sn = t.nm; g.nsn = t.n;
  g.cur = cfg_new_block(&g);
  g.exit_blk = cfg_new_block(&g);
  cfg_lower(&g, body);
  cfg_jump_to(&g, g.exit_blk);
  st_ssa_methods++;
  /* a local node the graph has no event for is a use it cannot place on
     a path: its name keeps one slot */
  for (int b = 0; b < g.nb; b++)
    for (int e = 0; e < g.b[b].nev; e++) if (g.b[b].ev_node[e] < g_ssa_rd_cap) g_ssa_seen[g.b[b].ev_node[e]] = 1;
  for (int q = 0; q < t.nnodes; q++)
    if (t.nodes[q] >= g_ssa_rd_cap || !g_ssa_seen[t.nodes[q]]) {
      if (ssa_log() && !t.bad[t.node_name[q]]) fprintf(stderr, "spinel-ssa:   `%s` at line %d has no place in the graph: one slot\n", t.nm[t.node_name[q]], ssa_line(c, t.nodes[q]));
      t.bad[t.node_name[q]] = 1;
    }
  for (int b = 0; b < g.nb; b++)
    for (int e = 0; e < g.b[b].nev; e++) if (g.b[b].ev_node[e] < g_ssa_rd_cap) g_ssa_seen[g.b[b].ev_node[e]] = 0;

  int nn = t.n;
  /* definitions: one entry definition per name (the argument, or nil
     before any write), then one per write event */
  int ndef = nn;
  for (int b = 0; b < g.nb; b++)
    for (int e = 0; e < g.b[b].nev; e++) if (g.b[b].ev_kind[e] != EV_READ) ndef++;
  int *def_name = malloc(sizeof(int) * (size_t)(ndef + 1));
  int *def_node = malloc(sizeof(int) * (size_t)(ndef + 1));
  int *uf = malloc(sizeof(int) * (size_t)(ndef + 1));
  char *has_read = calloc((size_t)ndef + 1, 1);
  if (!def_name || !def_node || !uf || !has_read) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  for (int i = 0; i < nn; i++) { def_name[i] = i; def_node[i] = -1; }
  int d = nn;
  for (int b = 0; b < g.nb; b++) {
    CBlock *bl = &g.b[b];
    bl->ev_def = malloc(sizeof(int) * (size_t)(bl->nev + 1));
    if (!bl->ev_def) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
    for (int e = 0; e < bl->nev; e++) {
      bl->ev_def[e] = -1;
      if (bl->ev_kind[e] == EV_READ) continue;
      int li = cfg_local_index(&g, nt_str(nt, bl->ev_node[e], "name"));
      def_name[d] = li; def_node[d] = bl->ev_node[e];
      bl->ev_def[e] = d++;
    }
  }
  for (int i = 0; i < ndef; i++) uf[i] = i;
  /* each name's definitions */
  int *dl_off = calloc((size_t)nn + 1, sizeof(int)), *dl = malloc(sizeof(int) * (size_t)(ndef + 1));
  if (!dl_off || !dl) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  for (int i = 0; i < ndef; i++) if (def_name[i] >= 0) dl_off[def_name[i] + 1]++;
  for (int i = 0; i < nn; i++) dl_off[i + 1] += dl_off[i];
  { int *fill = malloc(sizeof(int) * (size_t)(nn + 1));
    if (!fill) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
    memcpy(fill, dl_off, sizeof(int) * (size_t)nn);
    for (int i = 0; i < ndef; i++) if (def_name[i] >= 0) dl[fill[def_name[i]]++] = i;
    free(fill); }

  /* reaching definitions: a bit per definition, per block entry */
  int W = (ndef + 63) / 64, nb = g.nb;
  if ((size_t)nb * (size_t)W > ((size_t)1 << 22)) goto done;   /* a huge method: left as it is */
  {
    uint64_t *in = calloc((size_t)nb * (size_t)W + 1, sizeof(uint64_t));
    uint64_t *cur = malloc(sizeof(uint64_t) * ((size_t)W + 1));
    char *inq = calloc((size_t)nb + 1, 1), *reached = calloc((size_t)nb + 1, 1);
    int *queue = malloc(sizeof(int) * ((size_t)nb + 1));
    if (!in || !cur || !inq || !reached || !queue) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
    for (int i = 0; i < nn; i++) in[i / 64] |= (uint64_t)1 << (i % 64);
    /* the queue is a ring: a block is in it at most once */
    int qh = 0, qn = 0;
    queue[0] = 0; qn = 1; inq[0] = 1; reached[0] = 1;
#define SSA_TRANSFER(bi, rec) do { CBlock *_bl = &g.b[bi]; \
      for (int _e = 0; _e < _bl->nev; _e++) { \
        int _li = cfg_local_index(&g, nt_str(nt, _bl->ev_node[_e], "name")); \
        if (_li < 0) continue; \
        if (_bl->ev_kind[_e] == EV_READ) { if (rec) ssa_read(&g, _bl->ev_node[_e], _li, cur, dl, dl_off, uf, def_node); continue; } \
        for (int _q = dl_off[_li]; _q < dl_off[_li + 1]; _q++) cur[dl[_q] / 64] &= ~((uint64_t)1 << (dl[_q] % 64)); \
        int _dd = _bl->ev_def[_e]; cur[_dd / 64] |= (uint64_t)1 << (_dd % 64); \
      } } while (0)
    while (qn > 0) {
      int bi = queue[qh]; qh = (qh + 1) % nb; qn--; inq[bi] = 0;
      memcpy(cur, in + (size_t)bi * W, sizeof(uint64_t) * (size_t)W);
      SSA_TRANSFER(bi, 0);
      CBlock *bl = &g.b[bi];
      for (int k = 0; k < bl->nsucc; k++) {
        int sb = bl->succ[k];
        uint64_t *si = in + (size_t)sb * W;
        int changed = !reached[sb];
        reached[sb] = 1;
        for (int w = 0; w < W; w++) { uint64_t nv = si[w] | cur[w]; if (nv != si[w]) { si[w] = nv; changed = 1; } }
        if (changed && !inq[sb]) { queue[(qh + qn) % nb] = sb; qn++; inq[sb] = 1; }
      }
    }
    /* the final pass: every read joins the definitions it sees */
    for (int bi = 0; bi < nb; bi++) {
      if (!reached[bi]) continue;
      memcpy(cur, in + (size_t)bi * W, sizeof(uint64_t) * (size_t)W);
      SSA_TRANSFER(bi, 1);
    }
#undef SSA_TRANSFER
    free(in); free(cur); free(inq); free(reached); free(queue);
  }
  /* `x += 1`, `x ||= v`: one node reads and writes, so one name */
  for (int b = 0; b < g.nb; b++) {
    CBlock *bl = &g.b[b];
    for (int e = 0; e < bl->nev; e++) {
      int dd = bl->ev_def[e], node = bl->ev_node[e];
      if (dd < 0) continue;
      NodeKind nk = nt_kind(nt, node);
      if (nk != NK_LocalVariableOperatorWriteNode && nk != NK_LocalVariableOrWriteNode &&
          nk != NK_LocalVariableAndWriteNode) continue;
      if (g_ssa_rd[node] >= 0) ssa_union(uf, dd, g_ssa_rd[node]);
    }
  }
  /* a write whose value reads the local: the write joins the web of the
     writes that read sees (ssa_value_reads) */
  if (t.np > 0) {
    for (int b = 0; b < g.nb; b++)
      for (int e = 0; e < g.b[b].nev; e++)
        if (g.b[b].ev_def[e] >= 0 && g.b[b].ev_node[e] < g_ssa_rd_cap) g_ssa_wd[g.b[b].ev_node[e]] = g.b[b].ev_def[e];
    for (int q = 0; q < t.np; q++) {
      int w = t.pw[q] < g_ssa_rd_cap ? g_ssa_wd[t.pw[q]] : -1;
      int r = t.pr[q] < g_ssa_rd_cap ? g_ssa_rd[t.pr[q]] : -1;
      if (w >= 0 && r >= 0) ssa_union(uf, w, r);
    }
    for (int b = 0; b < g.nb; b++)
      for (int e = 0; e < g.b[b].nev; e++)
        if (g.b[b].ev_node[e] < g_ssa_rd_cap) g_ssa_wd[g.b[b].ev_node[e]] = -1;
  }
  for (int b = 0; b < g.nb; b++)
    for (int e = 0; e < g.b[b].nev; e++)
      if (g.b[b].ev_kind[e] == EV_READ && g_ssa_rd[g.b[b].ev_node[e]] >= 0)
        has_read[ssa_find(uf, g_ssa_rd[g.b[b].ev_node[e]])] = 1;

  /* the webs of each name, and their names */
  {
    char **web_name = calloc((size_t)ndef + 1, sizeof(char *));
    char *keeps = calloc((size_t)ndef + 1, 1);
    if (!web_name || !keeps) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
    int renamed_any = 0;
    for (int li = 0; li < nn; li++) {
      if (t.bad[li]) continue;
      char key[512];
      ssa_split_key(c, s, t.nm[li], key, sizeof key);
      if (ssa_vetoed(key)) { t.bad[li] = 1; continue; }
      int roots = 0, first_root = -1;
      for (int q = dl_off[li]; q < dl_off[li + 1]; q++) {
        int r = ssa_find(uf, dl[q]);
        int seen = 0;
        for (int q2 = dl_off[li]; q2 < q; q2++) if (ssa_find(uf, dl[q2]) == r) { seen = 1; break; }
        if (!seen) roots++;
      }
      if (roots < 2) continue;
      /* the entry definition's web keeps the name (a parameter is that
         name); when no read sees it, the first write's web joins it */
      int r0 = ssa_find(uf, li);
      keeps[r0] = 1;
      if (!has_read[r0]) {
        int best = -1;
        for (int q = dl_off[li]; q < dl_off[li + 1]; q++)
          if (def_node[dl[q]] >= 0 && (best < 0 || def_node[dl[q]] < def_node[best])) best = dl[q];
        if (best >= 0) { first_root = ssa_find(uf, best); keeps[first_root] = 1; }
      }
      int k = 0;
      for (int q = dl_off[li]; q < dl_off[li + 1]; q++) {
        int r = ssa_find(uf, dl[q]);
        if (keeps[r] || web_name[r]) continue;
        size_t ln = strlen(t.nm[li]) + 24;
        char *nmb = malloc(ln);
        if (!nmb) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
        for (;;) {
          snprintf(nmb, ln, "%s__ssa%d", t.nm[li], ++k);
          int clash = 0;
          for (int i = 0; i < nn; i++) if (sp_streq(t.nm[i], nmb)) { clash = 1; break; }
          if (!clash) break;
        }
        web_name[r] = nmb;
      }
      if (!k) continue;   /* every web keeps the name */
      st_ssa_split++;
      renamed_any = 1;
      if (g_nssa_splits >= g_cssa_splits) {
        g_cssa_splits = g_cssa_splits ? g_cssa_splits * 2 : 16;
        g_ssa_splits = realloc(g_ssa_splits, sizeof(SsaSplit) * (size_t)g_cssa_splits);
        if (!g_ssa_splits) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
      }
      {
        SsaSplit *sp = &g_ssa_splits[g_nssa_splits++];
        sp->scope = s->idx; sp->key = strdup(key);
        sp->names = malloc(sizeof(char *) * (size_t)(k + 2)); sp->nnames = 0;
        if (!sp->key || !sp->names) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
        sp->names[sp->nnames++] = strdup(t.nm[li]);
        for (int q = dl_off[li]; q < dl_off[li + 1]; q++) {
          int r = ssa_find(uf, dl[q]);
          if (!web_name[r]) continue;
          int dup = 0;
          for (int z = 1; z < sp->nnames; z++) if (sp_streq(sp->names[z], web_name[r])) { dup = 1; break; }
          if (!dup) sp->names[sp->nnames++] = strdup(web_name[r]);
        }
      }
      if (ssa_log()) {
        const char *cn = s->class_id >= 0 && s->class_id < c->nclasses ? c->classes[s->class_id].name : NULL;
        fprintf(stderr, "spinel-ssa: %s%s%s: `%s` splits into %d slots:", cn ? cn : "", cn ? "#" : "", s->name ? s->name : "?", t.nm[li], k + 1);
        int printed_keep = 0;
        for (int pass = 0; pass < 2; pass++)
          for (int q = dl_off[li]; q < dl_off[li + 1]; q++) {
            int r = ssa_find(uf, dl[q]);
            if (pass == 0 && !keeps[r]) continue;
            if (pass == 1 && keeps[r]) continue;
            int dup = 0;
            for (int q2 = dl_off[li]; q2 < q; q2++) if (ssa_find(uf, dl[q2]) == r) { dup = 1; break; }
            if (dup) continue;
            fprintf(stderr, " %s%s(", pass == 0 && printed_keep ? "+" : "", keeps[r] ? t.nm[li] : web_name[r]);
            int any = 0;
            for (int q2 = dl_off[li]; q2 < dl_off[li + 1]; q2++)
              if (ssa_find(uf, dl[q2]) == r) {
                fprintf(stderr, "%s%s", any ? "," : "", def_node[dl[q2]] < 0 ? "entry" : "");
                if (def_node[dl[q2]] >= 0) fprintf(stderr, "line %d", ssa_line(c, def_node[dl[q2]]));
                any = 1;
              }
            fprintf(stderr, ")");
            if (pass == 0) printed_keep = 1;
          }
        fprintf(stderr, "\n");
      }
    }
    /* rename the tree */
    if (renamed_any) {
      for (int b = 0; b < g.nb; b++) {
        CBlock *bl = &g.b[b];
        for (int e = 0; e < bl->nev; e++) {
          int node = bl->ev_node[e];
          int li = cfg_local_index(&g, nt_str(nt, node, "name"));
          if (li < 0 || t.bad[li]) continue;
          int dd = bl->ev_def[e] >= 0 ? bl->ev_def[e] : g_ssa_rd[node];
          if (dd < 0) continue;
          char *nmn = web_name[ssa_find(uf, dd)];
          if (!nmn || sp_streq(nt_str(nt, node, "name"), nmn)) continue;
          nt_set_str((NodeTable *)nt, node, "name", nmn);
          st_ssa_renamed++;
        }
      }
    }
    /* the def's `locals` list names its slots: the new ones join it */
    if (renamed_any) {
      const char *ol = nt_str(nt, s->def_node, "locals");
      size_t tot = (ol ? strlen(ol) : 0) + 1;
      for (int i = 0; i < ndef; i++) if (web_name[i]) tot += strlen(web_name[i]) + 1;
      char *nl = malloc(tot + 1);
      if (!nl) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
      snprintf(nl, tot + 1, "%s", ol ? ol : "");
      for (int i = 0; i < ndef; i++)
        if (web_name[i]) { if (nl[0]) strcat(nl, ","); strcat(nl, web_name[i]); }
      if (ol) nt_set_str((NodeTable *)nt, s->def_node, "locals", nl);
      free(nl);
    }
    for (int i = 0; i < ndef; i++) free(web_name[i]);
    free(web_name); free(keeps);
  }
done:
  for (int b = 0; b < g.nb; b++) {
    CBlock *bl = &g.b[b];
    for (int e = 0; e < bl->nev; e++) if (bl->ev_node[e] < g_ssa_rd_cap) g_ssa_rd[bl->ev_node[e]] = -1;
    free(bl->ev_kind); free(bl->ev_node); free(bl->succ); free(bl->ev_def);
  }
  free(g.b);
  free(def_name); free(def_node); free(uf); free(has_read); free(dl_off); free(dl);
  for (int i = 0; i < t.n; i++) free((char *)t.nm[i]);
  free(t.nm); free(t.bad); free(t.nodes); free(t.node_name); free(t.pw); free(t.pr);
}

int ssa_rename_locals(Compiler *c) {
  if (ssa_disabled()) return 0;
  const NodeTable *nt = c->nt;
  g_ssa_rd_cap = nt->count;
  g_ssa_rd = malloc(sizeof(int) * ((size_t)g_ssa_rd_cap + 1));
  if (!g_ssa_rd) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  g_ssa_wd = malloc(sizeof(int) * ((size_t)g_ssa_rd_cap + 1));
  if (!g_ssa_wd) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  for (int i = 0; i <= g_ssa_rd_cap; i++) g_ssa_rd[i] = g_ssa_wd[i] = -1;
  g_ssa_seen = calloc((size_t)g_ssa_rd_cap + 1, 1);
  if (!g_ssa_seen) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  st_ssa_methods = st_ssa_split = st_ssa_renamed = st_ssa_phis = 0;
  for (int si = 0; si < c->nscopes; si++) {
    Scope *s = &c->scopes[si];
    if (s->def_node < 0 || nt_kind(nt, s->def_node) != NK_DefNode || s->cs_synth) continue;
    /* the prelude's methods look some of their locals up by name */
    if (nt_int(nt, s->def_node, "node_bi", 0) || nt_int(nt, s->def_node, "node_line", 0) == 0) continue;
    ssa_method(c, s);
  }
  free(g_ssa_rd); g_ssa_rd = NULL; g_ssa_rd_cap = 0;
  free(g_ssa_wd); g_ssa_wd = NULL;
  free(g_ssa_seen); g_ssa_seen = NULL;
  if (ssa_log()) fprintf(stderr, "spinel-ssa: methods=%d phis=%d locals-split=%d nodes-renamed=%d\n",
                         st_ssa_methods, st_ssa_phis, st_ssa_split, st_ssa_renamed);
  return st_ssa_renamed;
}

/* The types the analysis gave the slots of each split. A split whose
   slots are all plain values stays. Any other (an Array, a Hash, a poly
   slot) is undone: the compile runs again from the start with those
   locals left whole, so what Spinel prints for them is what master prints.
   Called at the end of the analysis, before any C is written. */
static int ssa_plain(TyKind t) {
  return t == TY_NIL || t == TY_INT || t == TY_BIGINT || t == TY_FLOAT || t == TY_STRING ||
         t == TY_STRBUF || t == TY_SYMBOL || t == TY_BOOL;
}
void ssa_check_splits(Compiler *c) {
  if (g_nssa_splits == 0) return;
  size_t cap = 256, len = 0;
  char *veto = malloc(cap);
  if (!veto) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  veto[0] = 0;
  const char *old = getenv("SP_SSA_VETO");
  if (old && *old) {
    len = strlen(old);
    while (len + 2 > cap) cap *= 2;
    veto = realloc(veto, cap);
    if (!veto) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
    memcpy(veto, old, len + 1);
  }
  int nveto = 0;
  for (int i = 0; i < g_nssa_splits; i++) {
    SsaSplit *sp = &g_ssa_splits[i];
    Scope *sc = sp->scope >= 0 && sp->scope < c->nscopes ? &c->scopes[sp->scope] : NULL;
    if (!sc || !sc->reachable) continue;
    const char *why = NULL;
    for (int q = 0; q < sp->nnames && !why; q++) {
      LocalVar *lv = scope_local(sc, sp->names[q]);
      if (!lv || !ssa_plain(lv->type)) why = sp->names[q];
    }
    if (!why) continue;
    if (ssa_log()) {
      LocalVar *lv = scope_local(sc, why);
      fprintf(stderr, "spinel-ssa: %s: slot `%s` is %s, not a plain value: one slot again\n",
              sp->key, why, lv ? ty_name(lv->type) : "missing");
    }
    size_t kl = strlen(sp->key);
    while (len + kl + 2 > cap) cap *= 2;
    veto = realloc(veto, cap);
    if (!veto) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
    if (len) veto[len++] = ',';
    memcpy(veto + len, sp->key, kl + 1); len += kl;
    nveto++;
  }
  if (nveto == 0) { free(veto); return; }
  /* each round leaves more locals whole; a third round turns SSA off */
  const char *rs = getenv("SP_SSA_ROUND");
  int round = rs ? atoi(rs) : 0;
  char rb[16]; snprintf(rb, sizeof rb, "%d", round + 1);
  setenv("SP_SSA_ROUND", rb, 1);
  setenv("SP_SSA_VETO", veto, 1);
  if (round + 1 >= 3) setenv("SP_NO_SSA", "1", 1);
  free(veto);
  if (!g_sp_argv) return;   /* (no argv to run again: keep the splits) */
  if (ssa_log()) fprintf(stderr, "spinel-ssa: compiling again with %d local(s) left whole\n", nveto);
  fflush(stdout); fflush(stderr);
  execv("/proc/self/exe", g_sp_argv);
  fprintf(stderr, "spinel: cannot run the compile again for SSA\n");
  exit(1);
}
