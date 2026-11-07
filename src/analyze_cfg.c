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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { EV_READ = 1, EV_WRITE, EV_WRITE_UNK, EV_ANDWRITE, EV_OPWRITE };

typedef struct {
  int *ev_kind, *ev_node; int nev, cev;
  int *succ; int nsucc, csucc;
  int reached;
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
  for (int r = rescue; r >= 0; r = nt_ref(nt, r, "subsequent")) {
    int rb = cfg_new_block(g);
    cfg_add_edge(g, entry, rb);
    for (int k = first; k < last; k++) if (k != en) cfg_add_edge(g, k, rb);
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
  case NK_MultiWriteNode: case NK_MultiTargetNode: {
    cfg_lower(g, nt_ref(nt, n, "value"));
    const char *keys[] = { "lefts", "rights" };
    for (int q = 0; q < 2; q++) {
      int ln = 0; const int *l = nt_arr(nt, n, keys[q], &ln);
      for (int i = 0; i < ln; i++) cfg_lower(g, l[i]);
    }
    int r = nt_ref(nt, n, "rest");
    if (r >= 0) cfg_lower(g, nt_kind(nt, r) == NK_SplatNode ? nt_ref(nt, r, "expression") : r);
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
        int cn = 0; const int *cv = nt_arr(nt, w[i], "conditions", &cn);
        for (int q = 0; q < cn; q++) cfg_lower(g, cv[q]);
        cfg_add_edge(g, g->cur, arm); cfg_add_edge(g, g->cur, next);
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
    if (nt_safe_nav(nt, n)) {
      /* `a&.m(x = v)`: the arguments run only when a is not nil */
      int before = g->cur;
      int join = cfg_new_block(g);
      cfg_fall(g);
      cfg_lower(g, nt_ref(nt, n, "arguments"));
      cfg_jump_to(g, join);
      if (before >= 0) cfg_add_edge(g, before, join);
      g->cur = join;
    } else cfg_lower(g, nt_ref(nt, n, "arguments"));
    /* the block is its own scope; a bare raise/exit that reaches Kernel's
       never returns (a method of the program with the name does) */
    if (an_call_diverges(g->c, n)) cfg_jump_out(g, g->exit_blk, 0);
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
  for (int i = 0; i < g.nb; i++) { free(g.b[i].ev_kind); free(g.b[i].ev_node); free(g.b[i].succ); }
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
    comp_set_nilnarrow(c, r, (TyKind)fc_ty[r]);
    st_narrowed++;
  }
  free(par);
  if (cfg_log()) fprintf(stderr, "spinel-cfg: narrowed=%d\n", st_narrowed);
  return st_narrowed;
}
