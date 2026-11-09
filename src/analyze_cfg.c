/* analyze_cfg.c -- a per-method control-flow graph and one forward dataflow
   pass over it (prototype, see SP_NO_CFG).

   A method body is cut into basic blocks: runs of events (a local read, a
   local write) with no jump inside. Edges follow if/unless/case, and/or,
   while/until/for (with their back edges), begin/rescue/ensure (every
   protected block reaches each rescue arm), return/break/next/redo/retry and
   the diverging Kernel calls (raise, exit ...). A block, lambda, def or class
   body is a scope of its own and is not entered: a local a closure writes is
   left out of the facts altogether.

   On the graph sits an SSA IR (ir_build): a value for each write and for
   the method's entry, a phi where two values of a local meet, and the one
   value each read sees. Two facts per value, settled over the phis:
   - unset: the read may run before any write of the local (OR at a phi),
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

enum { EV_READ = 1, EV_WRITE, EV_WRITE_UNK, EV_ANDWRITE, EV_OPWRITE, EV_PI };

/* The kinds of value a local can hold at a point, as a set (the type
   lattice of the narrowing passes). KN_OBJ goes with one class id when
   all the objects are of one class. */
enum { KN_NIL = 1 << 0, KN_TRUE = 1 << 1, KN_FALSE = 1 << 2, KN_INT = 1 << 3, KN_BIG = 1 << 4,
       KN_FLOAT = 1 << 5, KN_STR = 1 << 6, KN_SYM = 1 << 7, KN_ARR = 1 << 8, KN_HASH = 1 << 9,
       KN_OBJ = 1 << 10, KN_OTHER = 1 << 11 };
#define KN_ALL 0x0fff
/* The IR's own types (ir_own_kinds) mark a value they know nothing of
   with KN_UNK too, so a test that narrows it (`x || 1`) keeps it unknown */
#define KN_UNK (1 << 12)
#define KN_INTS (KN_INT | KN_BIG)
#define KN_BOOL (KN_TRUE | KN_FALSE)
#define KN_BOXY (KN_ARR | KN_HASH | KN_OBJ | KN_OTHER)

typedef struct {
  int *ev_kind, *ev_node; int nev, cev;
  int *succ; int nsucc, csucc;
  int reached;
  int *ev_val;       /* the IR's value: the one a read sees, the one a write
                        or a pi makes (ir_build) */
  int *ev_loc;       /* the local's index (cfg_local_index), or -1 */
  int *ev_aux;       /* a pi event: its index in CfgB.pis, else -1 */
} CBlock;

/* A pi: what an edge out of a type test tells about the local it tests.
   `if v.is_a?(String)` puts PI_ISA on the true edge and PI_NOTISA on the
   false one; `if v` puts PI_TRUTHY and PI_FALSY. ZJIT's HIR writes the
   same thing as a GuardType, with a side exit; here the program's own
   test is the guard, so no exit is needed. */
enum { PI_ISA = 1, PI_NOTISA, PI_NIL, PI_NOTNIL, PI_TRUTHY, PI_FALSY };
typedef struct {
  int test;              /* the test's node */
  int op;                /* PI_* */
  unsigned short mask;   /* PI_ISA/PI_NOTISA: the class's kinds (KN_*) */
  short cid;             /* a user class: its index, else -1 */
  char loose;            /* an instance of a subclass passes the test too */
  const char *cls;       /* the class's name, for the dump */
} CfgPi;

typedef struct {
  Compiler *c;
  const NodeTable *nt;
  Scope *s;
  int si;
  /* full mode (the narrowing passes): a pi on each edge out of a type
     test, conditions lowered as jumps, and the blocks of calls on the
     graph. `blocks` alone (the SSA rename) puts the blocks there. */
  int full, blocks;
  CfgPi *pis; int npis, cpis;
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
static int cfg_local_index(CfgB *g, const char *nm);
static int cfg_subtree_writes(const NodeTable *nt, int n, const char *nm);
static void cfg_push_event(CfgB *g, CBlock *bl, int kind, int node, int loc, int aux) {
  (void)g;
  if (bl->nev >= bl->cev) {
    bl->cev = bl->cev ? bl->cev * 2 : 8;
    bl->ev_kind = realloc(bl->ev_kind, sizeof(int) * (size_t)bl->cev);
    bl->ev_node = realloc(bl->ev_node, sizeof(int) * (size_t)bl->cev);
    bl->ev_loc = realloc(bl->ev_loc, sizeof(int) * (size_t)bl->cev);
    bl->ev_aux = realloc(bl->ev_aux, sizeof(int) * (size_t)bl->cev);
    bl->ev_val = realloc(bl->ev_val, sizeof(int) * (size_t)bl->cev);
    if (!bl->ev_kind || !bl->ev_node || !bl->ev_loc || !bl->ev_aux || !bl->ev_val) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  }
  bl->ev_kind[bl->nev] = kind; bl->ev_node[bl->nev] = node;
  bl->ev_loc[bl->nev] = loc; bl->ev_aux[bl->nev] = aux; bl->ev_val[bl->nev] = -1;
  bl->nev++;
}
static void cfg_emit(CfgB *g, int kind, int node) {
  /* inside a protected body every write starts a new block, so the state
     before it is a block exit an exception edge carries to the rescue */
  if (g->protect > 0 && kind != EV_READ) cfg_fall(g);
  if (g->cur < 0) g->cur = cfg_new_block(g);   /* dead code: a block no edge reaches */
  cfg_push_event(g, &g->b[g->cur], kind, node, cfg_local_index(g, nt_str(g->nt, node, "name")), -1);
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
static void cfg_lower_list(CfgB *g, int stmts);
static void cfg_jump_to(CfgB *g, int target);

/* ---- full mode: pis, conditions as jumps, blocks on the graph ---- */

int class_has_subclass(Compiler *c, int ocid);

/* a user class inherits from the builtin class `name` (so an is_a? of it
   passes for an object of that class too) */
static int cfg_builtin_subclassed(Compiler *c, const char *name) {
  static Compiler *mc; static int mn = -1, memo[16];
  static const char *const names[] = { "String", "Symbol", "Integer", "Float", "Array", "Hash",
                                       "NilClass", "TrueClass", "FalseClass", NULL };
  int slot = -1;
  for (int i = 0; names[i]; i++) if (sp_streq(names[i], name)) { slot = i; break; }
  if (mc != c || mn != c->nclasses) { mc = c; mn = c->nclasses; for (int i = 0; i < 16; i++) memo[i] = -1; }
  if (slot >= 0 && memo[slot] >= 0) return memo[slot];
  int r = 0;
  for (int ci = 0; ci < c->nclasses && !r; ci++)
    for (int a = ci, d = 0; a >= 0 && d < 64 && !r; a = c->classes[a].parent, d++) {
      int dn = c->classes[a].def_node;
      int sup = dn >= 0 && nt_kind(c->nt, dn) == NK_ClassNode ? nt_ref(c->nt, dn, "superclass") : -1;
      if (sup >= 0 && nt_kind(c->nt, sup) == NK_ConstantReadNode && sp_streq(nt_str(c->nt, sup, "name"), name)) r = 1;
      if (a != ci && c->classes[a].name && sp_streq(c->classes[a].name, name)) r = 1;
    }
  if (slot >= 0) memo[slot] = r;
  return r;
}
/* What `is_a?(name)` passes: the kinds of a builtin value class, or the
   objects of a user class (not a module). 0 when the test says nothing
   the lattice can hold. */
static int cfg_class_kinds(Compiler *c, const char *name, unsigned short *mask, int *cid, char *loose) {
  *mask = 0; *cid = -1; *loose = 0;
  if (!name) return 0;
  static const struct { const char *n; unsigned short m; } B[] = {
    { "String", KN_STR }, { "Symbol", KN_SYM }, { "Integer", KN_INT | KN_BIG }, { "Float", KN_FLOAT },
    { "Array", KN_ARR }, { "Hash", KN_HASH }, { "NilClass", KN_NIL }, { "TrueClass", KN_TRUE },
    { "FalseClass", KN_FALSE }, { NULL, 0 } };
  for (int i = 0; B[i].n; i++)
    if (sp_streq(name, B[i].n)) { *mask = B[i].m; *loose = (char)cfg_builtin_subclassed(c, name); return 1; }
  if (is_builtin_class_name(name) || is_builtin_module_name(name)) return 0;
  int ci = comp_class_index(c, name);
  if (ci < 0) return 0;
  const ClassInfo *k = &c->classes[ci];
  if (!class_def_is_class(c->nt, k->def_node) || k->is_builtin_const || k->is_native_class || k->ary_root > 0)
    return 0;
  *mask = KN_OBJ; *cid = ci; *loose = (char)class_has_subclass(c, ci);
  return 1;
}
static int cfg_new_pi(CfgB *g, int test, int op, unsigned short mask, int cid, char loose, const char *cls) {
  if (g->npis >= g->cpis) {
    g->cpis = g->cpis ? g->cpis * 2 : 16;
    g->pis = realloc(g->pis, sizeof(CfgPi) * (size_t)g->cpis);
    if (!g->pis) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  }
  CfgPi *p = &g->pis[g->npis];
  p->test = test; p->op = op; p->mask = mask; p->cid = (short)(cid > 32767 ? -1 : cid); p->loose = loose; p->cls = cls;
  if (cid > 32767) p->loose = 1;   /* (an id past the field: an object of some class) */
  return g->npis++;
}
/* the edge from -> to, through a fresh block that holds a pi of local li */
static void cfg_edge_pi(CfgB *g, int from, int to, int li, int pi) {
  if (from < 0) return;
  if (li < 0 || pi < 0) { cfg_add_edge(g, from, to); return; }
  int e = cfg_new_block(g);
  cfg_add_edge(g, from, e);
  cfg_push_event(g, &g->b[e], EV_PI, g->pis[pi].test, li, pi);
  cfg_add_edge(g, e, to);
}
/* the pis of `is_a?(K)` (or `K === x`) on its two edges */
static void cfg_class_pis(CfgB *g, int test, int knode, int exact, int *pt, int *pf) {
  *pt = *pf = -1;
  if (knode < 0 || nt_kind(g->nt, knode) != NK_ConstantReadNode) return;
  const char *cn = nt_str(g->nt, knode, "name");
  unsigned short mask; int cid; char loose;
  if (!cfg_class_kinds(g->c, cn, &mask, &cid, &loose)) return;
  /* a class with a `self.===` of the program's own answers a when as it likes */
  int ci = comp_class_index(g->c, cn);
  if (ci >= 0 && class_any_cmethod_eqq(g->c, ci) >= 0) return;
  *pt = cfg_new_pi(g, test, PI_ISA, mask, cid, loose, cn);
  if (!exact) *pf = cfg_new_pi(g, test, PI_NOTISA, mask, cid, loose, cn);
}
/* the local a condition tests, and the pis of its true and false edges:
   `x`, `x.nil?`, `x.is_a?(K)`, `x.kind_of?(K)`, `x.instance_of?(K)`,
   `K === x`. -1 when it tests no local. */
static int cfg_test_of(CfgB *g, int pred, int *pt, int *pf) {
  const NodeTable *nt = g->nt;
  *pt = *pf = -1;
  if (pred < 0) return -1;
  NodeKind k = nt_kind(nt, pred);
  if (k == NK_LocalVariableReadNode) {
    if (!cfg_owned(g, pred)) return -1;
    int li = cfg_local_index(g, nt_str(nt, pred, "name"));
    if (li < 0) return -1;
    *pt = cfg_new_pi(g, pred, PI_TRUTHY, 0, -1, 0, NULL);
    *pf = cfg_new_pi(g, pred, PI_FALSY, 0, -1, 0, NULL);
    return li;
  }
  if (k != NK_CallNode || nt_safe_nav(nt, pred)) return -1;
  { int blk = nt_ref(nt, pred, "block"); if (blk >= 0) return -1; }
  const char *nm = nt_str(nt, pred, "name");
  if (!nm) return -1;
  int recv = nt_ref(nt, pred, "receiver");
  int args = nt_ref(nt, pred, "arguments");
  int an = 0; const int *av = args >= 0 ? nt_arr(nt, args, "arguments", &an) : NULL;
  /* `K === x` */
  if (sp_streq(nm, "===") && recv >= 0 && nt_kind(nt, recv) == NK_ConstantReadNode && an == 1 &&
      nt_kind(nt, av[0]) == NK_LocalVariableReadNode && cfg_owned(g, av[0])) {
    int li = cfg_local_index(g, nt_str(nt, av[0], "name"));
    if (li < 0) return -1;
    cfg_class_pis(g, pred, recv, 0, pt, pf);
    return *pt >= 0 ? li : -1;
  }
  if (recv < 0 || nt_kind(nt, recv) != NK_LocalVariableReadNode || !cfg_owned(g, recv)) return -1;
  int li = cfg_local_index(g, nt_str(nt, recv, "name"));
  if (li < 0) return -1;
  /* a method of the program's own with the name answers as it likes */
  if (an_user_defines_or_reads(g->c, nm)) return -1;
  if (sp_streq(nm, "nil?") && an == 0) {
    *pt = cfg_new_pi(g, pred, PI_NIL, 0, -1, 0, NULL);
    *pf = cfg_new_pi(g, pred, PI_NOTNIL, 0, -1, 0, NULL);
    return li;
  }
  if ((sp_streq(nm, "is_a?") || sp_streq(nm, "kind_of?") || sp_streq(nm, "instance_of?")) && an == 1) {
    cfg_class_pis(g, pred, av[0], sp_streq(nm, "instance_of?"), pt, pf);
    return *pt >= 0 ? li : -1;
  }
  return -1;
}
/* Lowers condition `pred` as jumps: to tb when it holds, to fb when not.
   `a && b` tests b only on a's true edge, `a || b` only on its false one,
   and `!a` swaps the two. Ends with no current block. */
static void cfg_cond(CfgB *g, int pred, int tb, int fb) {
  const NodeTable *nt = g->nt;
  int p = pred >= 0 ? nt_paren_single(nt, pred) : pred;
  NodeKind k = p >= 0 ? nt_kind(nt, p) : NK_NilNode;
  if (p >= 0 && (k == NK_AndNode || k == NK_OrNode)) {
    int mid = cfg_new_block(g);
    if (k == NK_AndNode) cfg_cond(g, nt_ref(nt, p, "left"), mid, fb);
    else cfg_cond(g, nt_ref(nt, p, "left"), tb, mid);
    g->cur = mid;
    cfg_cond(g, nt_ref(nt, p, "right"), tb, fb);
    return;
  }
  if (p >= 0 && k == NK_CallNode && nt_str(nt, p, "name") && sp_streq(nt_str(nt, p, "name"), "!") &&
      nt_ref(nt, p, "receiver") >= 0 && nt_ref(nt, p, "arguments") < 0 && nt_ref(nt, p, "block") < 0 &&
      !an_user_defines_or_reads(g->c, "!")) {
    cfg_cond(g, nt_ref(nt, p, "receiver"), fb, tb);
    return;
  }
  cfg_lower(g, pred);
  int from = g->cur;
  if (from < 0) return;
  int pt, pf;
  int li = cfg_test_of(g, p, &pt, &pf);
  cfg_edge_pi(g, from, tb, li, pt);
  cfg_edge_pi(g, from, fb, li, pf);
  g->cur = -1;
}
/* A write event of local li (for a name with no node of its own) */
static void cfg_emit_loc(CfgB *g, int kind, int node, int li) {
  if (g->protect > 0 && kind != EV_READ) cfg_fall(g);
  if (g->cur < 0) g->cur = cfg_new_block(g);
  cfg_push_event(g, &g->b[g->cur], kind, node, li, -1);
}
/* Whether the block of call `call` goes on the graph: it runs while the
   call runs, any number of times, and is done when the call returns. A
   block that can run later (a proc, a thread, a handler, a block a method
   keeps) is left off, as a closure. `typed`: the analysis has types, so
   the capture marking's own test applies; before them (the SSA rename)
   only the forms that always make a proc are left off, and the rename's
   final check undoes a split a later-running block reads. */
int an_block_outlives_call(Compiler *c, int id);
static int cfg_block_on_graph(Compiler *c, int call, int typed) {
  const NodeTable *nt = c->nt;
  NodeKind k = nt_kind(nt, call);
  if (k != NK_CallNode && k != NK_SuperNode && k != NK_ForwardingSuperNode) return 0;
  int blk = nt_ref(nt, call, "block");
  if (blk < 0 || nt_kind(nt, blk) != NK_BlockNode) return 0;
  if (k == NK_CallNode) {
    const char *nm = nt_str(nt, call, "name");
    if (!nm) return 0;
    if (is_proc_literal(c, call)) return 0;
    static const char *const later[] = { "at_exit", "trap", "define_method", "define_singleton_method",
      "instance_eval", "instance_exec", "class_eval", "class_exec", "module_eval", "module_exec",
      "define_finalizer", "fork", "spawn", "lazy", "to_enum", "enum_for", "set_trace_func", NULL };
    for (int i = 0; later[i]; i++) if (sp_streq(nm, later[i])) return 0;
    int recv = nt_ref(nt, call, "receiver");
    if (recv >= 0 && nt_kind(nt, recv) == NK_ConstantReadNode) {
      const char *rn = nt_str(nt, recv, "name");
      static const char *const makers[] = { "Thread", "Fiber", "Enumerator", "Hash", "Proc", "Class",
        "Module", "Struct", "Data", "Ractor", "TracePoint", "Signal", "ObjectSpace", "Process", NULL };
      for (int i = 0; rn && makers[i]; i++) if (sp_streq(rn, makers[i])) return 0;
    }
    /* `x.lazy.map { }` runs its block when the result is read */
    for (int r = recv, d = 0; r >= 0 && d < 16 && nt_kind(nt, r) == NK_CallNode; r = nt_ref(nt, r, "receiver"), d++)
      if (nt_str(nt, r, "name") && sp_streq(nt_str(nt, r, "name"), "lazy")) return 0;
  }
  if (typed && an_block_outlives_call(c, call)) return 0;
  return 1;
}
/* the names the reads under n read */
static void cfg_read_names(CfgB *g, int n, int depth, char *seen) {
  if (n < 0 || depth > 200) return;
  const NodeTable *nt = g->nt;
  NodeKind k = nt_kind(nt, n);
  if (k == NK_DefNode || k == NK_ClassNode || k == NK_ModuleNode || k == NK_SingletonClassNode) return;
  if (k == NK_LocalVariableReadNode) {
    int li = cfg_local_index(g, nt_str(nt, n, "name"));
    if (li >= 0 && !seen[li]) { seen[li] = 1; cfg_emit_loc(g, EV_WRITE_UNK, n, li); }
  }
  for (int i = 0; i < nt_num_refs(nt, n); i++) cfg_read_names(g, nt_ref_at(nt, n, i), depth + 1, seen);
  for (int i = 0; i < nt_num_arrs(nt, n); i++) {
    int an = 0; const int *av = nt_arr_at(nt, n, i, &an);
    for (int q = 0; q < an; q++) cfg_read_names(g, av[q], depth + 1, seen);
  }
}
/* A closure that can run at any later time (a proc, a lambda, a thread's
   body), in full mode: a region of its own, entered from where it is made,
   where each local it reads starts as any value. So only a test inside it
   narrows a read in it, and its writes (the closure marking leaves those
   locals out) reach nothing after it. */
static void cfg_lower_detached(CfgB *g, int n) {
  const NodeTable *nt = g->nt;
  int from = g->cur;
  if (from < 0) return;
  int entry = cfg_new_block(g), out = cfg_new_block(g);
  cfg_add_edge(g, from, entry);
  g->cur = entry;
  int nl = g->s->nlocals;
  char *seen = calloc((size_t)nl + 1, 1);
  if (!seen) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  cfg_read_names(g, n, 0, seen);
  free(seen);
  int again = cfg_fall(g);
  int sv_nlp = g->nlp, sv_nbg = g->nbg, sv_nen = g->nen, sv_prot = g->protect;
  g->nbg = 0; g->nen = 0; g->protect = 0;
  if (g->nlp < 64) { g->lp_head[g->nlp] = again; g->lp_body[g->nlp] = again; g->lp_exit[g->nlp] = out; g->lp_endepth[g->nlp] = 0; }
  g->nlp++;
  cfg_lower(g, nt_ref(nt, n, "parameters"));
  cfg_lower(g, nt_ref(nt, n, "body"));
  cfg_jump_to(g, out);
  g->nlp = sv_nlp; g->nbg = sv_nbg; g->nen = sv_nen; g->protect = sv_prot;
  g->cur = from;
}
/* A block on the graph: the call runs it any number of times, from a head
   block. Each run binds the block's own locals afresh (its parameters, nil
   for the rest), runs the defaults, then the body; `next` goes back to the
   head, `break` leaves, `redo` runs the body again on the same values. */
static void cfg_lower_block(CfgB *g, int blk) {
  const NodeTable *nt = g->nt;
  int head = cfg_new_block(g), bodyb = cfg_new_block(g), exitb = cfg_new_block(g);
  cfg_jump_to(g, head);
  cfg_add_edge(g, head, bodyb); cfg_add_edge(g, head, exitb);
  g->cur = bodyb;
  const char *ls = nt_str(nt, blk, "locals");
  for (const char *p = ls; p && *p; ) {
    const char *e = strchr(p, ',');
    size_t l = e ? (size_t)(e - p) : strlen(p);
    char nm[256];
    if (l > 0 && l < sizeof nm) {
      memcpy(nm, p, l); nm[l] = 0;
      int li = cfg_local_index(g, nm);
      if (li >= 0) cfg_emit_loc(g, EV_WRITE_UNK, blk, li);
    }
    if (!e) break;
    p = e + 1;
  }
  cfg_lower(g, nt_ref(nt, blk, "parameters"));
  int again = cfg_fall(g);
  if (g->nlp < 64) { g->lp_head[g->nlp] = head; g->lp_body[g->nlp] = again; g->lp_exit[g->nlp] = exitb; g->lp_endepth[g->nlp] = g->nen; }
  g->nlp++;
  cfg_lower(g, nt_ref(nt, blk, "body"));
  cfg_jump_to(g, head);
  g->nlp--;
  g->cur = exitb;
}

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
  if (g->full) {   /* `while x`, `until x.nil?`: a pi on each edge of the test */
    if (nt_kind(nt, n) == NK_UntilNode) cfg_cond(g, pred, exitb, bodyb);
    else cfg_cond(g, pred, bodyb, exitb);
  } else {
    cfg_lower(g, pred);
    if (g->cur >= 0) { cfg_add_edge(g, g->cur, bodyb); cfg_add_edge(g, g->cur, exitb); }
  }
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
  case NK_DefinedNode:
    return;
  case NK_BlockNode: case NK_LambdaNode:
    if (g->full) cfg_lower_detached(g, n);
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
    if (g->full && cfg_owned(g, n)) {
      /* the test is x's truthiness: x keeps its value on one edge, which
         the pi narrows; the other stores v */
      cfg_emit(g, EV_READ, n);
      int li = cfg_local_index(g, nt_str(nt, n, "name"));
      int from = g->cur;
      int join = cfg_new_block(g), asg = cfg_new_block(g);
      int pt = cfg_new_pi(g, n, PI_TRUTHY, 0, -1, 0, NULL), pf = cfg_new_pi(g, n, PI_FALSY, 0, -1, 0, NULL);
      int orw = k == NK_LocalVariableOrWriteNode;
      cfg_edge_pi(g, from, orw ? join : asg, li, pt);
      cfg_edge_pi(g, from, orw ? asg : join, li, pf);
      g->cur = asg;
      cfg_lower(g, nt_ref(nt, n, "value"));
      cfg_emit(g, EV_WRITE, n);
      cfg_jump_to(g, join);
      g->cur = join;
      return;
    }
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
    if (g->full && pred >= 0 && nt_kind(nt, nt_paren_single(nt, pred)) != NK_MatchPredicateNode) {
      /* the condition as jumps, with a pi on each edge out of a test */
      int then = nt_ref(nt, n, "statements");
      int els = nt_ref(nt, n, k == NK_IfNode ? "subsequent" : "else_clause");
      int join = cfg_new_block(g), pos = cfg_new_block(g), neg = cfg_new_block(g);
      cfg_cond(g, pred, pos, neg);
      g->cur = pos;
      if (k == NK_IfNode) cfg_lower_list(g, then); else if (els >= 0) cfg_lower(g, els);
      cfg_jump_to(g, join);
      g->cur = neg;
      if (k == NK_IfNode) { if (els >= 0) cfg_lower(g, els); } else cfg_lower_list(g, then);
      cfg_jump_to(g, join);
      g->cur = join;
      return;
    }
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
    if (g->full) {   /* `x.is_a?(K) && x.m`: the right side runs on one edge of the left's test */
      int side = cfg_new_block(g), join = cfg_new_block(g);
      if (k == NK_AndNode) cfg_cond(g, nt_ref(nt, n, "left"), side, join);
      else cfg_cond(g, nt_ref(nt, n, "left"), join, side);
      g->cur = side;
      cfg_lower(g, nt_ref(nt, n, "right"));
      cfg_jump_to(g, join);
      g->cur = join;
      return;
    }
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
    if (g->full && k == NK_CaseNode) {
      /* `case v; when String`: each condition is `String === v`, a class
         test of v, so its two edges take pis. No subject: each condition
         is a test of its own. */
      int subj = nt_ref(nt, n, "predicate");
      cfg_lower(g, subj);
      int sp = subj >= 0 ? nt_paren_single(nt, subj) : -1;
      int sl = -1;
      int wn = 0; const int *w = nt_arr(nt, n, "conditions", &wn);
      if (sp >= 0 && nt_kind(nt, sp) == NK_LocalVariableReadNode && cfg_owned(g, sp)) {
        const char *snm = nt_str(nt, sp, "name");
        sl = cfg_local_index(g, snm);
        /* (a condition that writes the subject tests the old value) */
        for (int i = 0; i < wn && sl >= 0; i++) {
          int cn = 0; const int *cv = nt_arr(nt, w[i], "conditions", &cn);
          for (int q = 0; q < cn; q++) if (cfg_subtree_writes(nt, cv[q], snm)) { sl = -1; break; }
        }
      }
      int join = cfg_new_block(g);
      int test = cfg_fall(g);
      for (int i = 0; i < wn; i++) {
        int next = cfg_new_block(g), arm = cfg_new_block(g);
        g->cur = test;
        int cn = 0; const int *cv = nt_arr(nt, w[i], "conditions", &cn);
        for (int q = 0; q < cn; q++) {
          int nx = q + 1 < cn ? cfg_new_block(g) : next;
          if (subj < 0) cfg_cond(g, cv[q], arm, nx);
          else {
            cfg_lower(g, cv[q]);
            int pt = -1, pf = -1;
            if (sl >= 0) {
              NodeKind ck = nt_kind(nt, cv[q]);
              if (ck == NK_ConstantReadNode) cfg_class_pis(g, cv[q], cv[q], 0, &pt, &pf);
              else if (ck == NK_NilNode) {
                pt = cfg_new_pi(g, cv[q], PI_NIL, 0, -1, 0, NULL);
                pf = cfg_new_pi(g, cv[q], PI_NOTNIL, 0, -1, 0, NULL);
              }
            }
            int from = g->cur;
            cfg_edge_pi(g, from, arm, sl, pt);
            cfg_edge_pi(g, from, nx, sl, pf);
          }
          g->cur = nx;
        }
        if (cn == 0) cfg_jump_to(g, next);
        g->cur = arm;
        cfg_lower_list(g, nt_ref(nt, w[i], "statements"));
        cfg_jump_to(g, join);
        test = next;
      }
      g->cur = test;
      int els = nt_ref(nt, n, "else_clause");
      if (els >= 0) cfg_lower(g, els);
      cfg_jump_to(g, join);
      g->cur = join;
      return;
    }
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
      if (blk >= 0 && nt_kind(nt, blk) != NK_BlockNode) cfg_lower(g, blk);
      else if (blk >= 0 && g->blocks && cfg_block_on_graph(g->c, n, g->full)) cfg_lower_block(g, blk);
      else if (blk >= 0 && g->full) cfg_lower_detached(g, blk); }
    if (join >= 0) {
      cfg_jump_to(g, join);
      if (before >= 0) cfg_add_edge(g, before, join);
      g->cur = join;
    }
    /* the block is its own scope (or, on the graph, runs inside the call);
       a bare raise/exit that reaches Kernel's never returns (a method of
       the program with the name does) */
    if (!g->ssa && an_call_diverges(g->c, n)) cfg_jump_out(g, g->exit_blk, 0);
    return;
  }
  case NK_SuperNode: case NK_ForwardingSuperNode:
    if (g->blocks) {   /* `super(x) { ... }`: the block runs inside the call */
      cfg_lower(g, nt_ref(nt, n, "arguments"));
      int blk = nt_ref(nt, n, "block");
      if (blk >= 0 && nt_kind(nt, blk) != NK_BlockNode) cfg_lower(g, blk);
      else if (blk >= 0 && cfg_block_on_graph(g->c, n, g->full)) cfg_lower_block(g, blk);
      else if (blk >= 0 && g->full) cfg_lower_detached(g, blk);
      return;
    }
    cfg_lower_children(g, n);
    return;
  default:
    cfg_lower_children(g, n);
    return;
  }
}

/* ---- the SSA IR: one value per write, a phi where values meet ----------

   Each write event makes a value, and so does the method's entry, once
   per local: the argument for a parameter, nil for any other local. A read
   sees exactly one value. Where paths with different values of a local
   meet, the block takes a phi: a value with one operand per predecessor.
   ZJIT's HIR writes the same thing as a block parameter, and each jump
   passes the operand for it.

   The phis are placed from the reads, on demand (Braun et al.): a read
   asks for its local's value at its block's entry; a block with one
   predecessor answers with that block's last write, or asks it in turn; a
   join block makes a phi and asks each predecessor for an operand. Then a
   phi whose operands are all one value (or the phi itself) stands for that
   value and is dropped, until none is left. A phi that no read reaches
   (through other phis) is dead: the passes skip it.

   ir_add_exprs (below) adds a value for each step of what a write stores:
   a Const per literal, a Send per call, a Phi where the arms of an `if` or
   a `case` join, an And or an Or for `&&` and `||`, a Rescued for the
   exception a rescue arm binds, and an Other for anything else. */
enum { IV_ENTRY = 1, IV_DEF, IV_PHI, IV_PI, IV_CONST, IV_SEND, IV_JOIN, IV_AND, IV_OR, IV_RESCUED, IV_OTHER };
/* a SEND's flags (its evk): a splat, keyword or block-pass argument; a
   block; `&.` */
enum { SF_ODD = 1, SF_BLOCK = 2, SF_SAFE = 4 };

typedef struct {
  unsigned char kind, live;
  int local;
  int node;        /* a DEF: the write's node; a PI: the test's node */
  int evk;         /* a DEF: the event kind (EV_WRITE ...); a PI: its CfgB.pis index */
  int block;
  int src;         /* a PI, and the DEF of `x &&= v`: the value before it */
  int op0, nop;    /* a PHI: ops[op0 .. op0 + nop), one per predecessor; a SEND: the
                      receiver (-1: self), then the arguments; a JOIN, an AND, an OR:
                      the values it joins */
  int repl;        /* a trivial PHI: the value it stands for, else -1 */
  int arg;         /* a DEF: the value it stores (ir_add_exprs), else -1 */
  int owner;       /* a value of what a write stores: that write's DEF, else -1 */
} IrVal;

typedef struct {
  CfgB *g;
  int nl, nb;
  IrVal *v; int nv, cv;
  int *ops; int nops, cops;
  /* a local's value at a block's entry, for the pairs asked so far: open
     addressing over key = block * nl + local */
  long long *ek; int *ev; int ecap, en;
  int *pred_off, *preds;
  char *reached, *phiblk;
  int *rpo_pos;
  int *chain;
  int *pend; int npend, cpend;
  int nphi, ntrivial;
} Ir;

static void *ir_xalloc(size_t n) {
  void *p = calloc(n ? n : 1, 1);
  if (!p) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  return p;
}
static int ir_val_new(Ir *ir, int kind, int local, int node, int evk, int block) {
  if (ir->nv >= ir->cv) {
    ir->cv = ir->cv ? ir->cv * 2 : 64;
    ir->v = realloc(ir->v, sizeof(IrVal) * (size_t)ir->cv);
    if (!ir->v) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  }
  IrVal *x = &ir->v[ir->nv];
  memset(x, 0, sizeof *x);
  x->kind = (unsigned char)kind; x->local = local; x->node = node; x->evk = evk; x->block = block;
  x->src = -1; x->op0 = -1; x->repl = -1; x->arg = -1; x->owner = -1;
  return ir->nv++;
}
static int ir_find(Ir *ir, int x) {
  if (x < 0) return x;
  int r = x;
  while (ir->v[r].repl >= 0) r = ir->v[r].repl;
  while (ir->v[x].repl >= 0) { int nx = ir->v[x].repl; ir->v[x].repl = r; x = nx; }
  return r;
}
static int ir_eget(Ir *ir, int b, int l) {
  if (ir->ecap == 0) return -1;
  long long key = (long long)b * ir->nl + l;
  unsigned h = (unsigned)((unsigned long long)key * 0x9E3779B97F4A7C15ULL >> 32) & (unsigned)(ir->ecap - 1);
  while (ir->ek[h] >= 0) { if (ir->ek[h] == key) return ir->ev[h]; h = (h + 1) & (unsigned)(ir->ecap - 1); }
  return -1;
}
static void ir_eset(Ir *ir, int b, int l, int val) {
  if ((ir->en + 1) * 2 > ir->ecap) {
    int oc = ir->ecap; long long *ok = ir->ek; int *ov = ir->ev;
    ir->ecap = oc ? oc * 2 : 256;
    ir->ek = malloc(sizeof(long long) * (size_t)ir->ecap); ir->ev = malloc(sizeof(int) * (size_t)ir->ecap);
    if (!ir->ek || !ir->ev) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
    for (int i = 0; i < ir->ecap; i++) ir->ek[i] = -1;
    ir->en = 0;
    for (int i = 0; i < oc; i++) if (ok[i] >= 0) ir_eset(ir, (int)(ok[i] / ir->nl), (int)(ok[i] % ir->nl), ov[i]);
    free(ok); free(ov);
  }
  long long key = (long long)b * ir->nl + l;
  unsigned h = (unsigned)((unsigned long long)key * 0x9E3779B97F4A7C15ULL >> 32) & (unsigned)(ir->ecap - 1);
  while (ir->ek[h] >= 0 && ir->ek[h] != key) h = (h + 1) & (unsigned)(ir->ecap - 1);
  if (ir->ek[h] < 0) { ir->ek[h] = key; ir->en++; }
  ir->ev[h] = val;
}
/* the value a write or a pi in block b leaves for local l, or -1 */
static int ir_last_write(Ir *ir, int b, int l) {
  CBlock *bl = &ir->g->b[b];
  for (int e = bl->nev - 1; e >= 0; e--)
    if (bl->ev_loc[e] == l && bl->ev_kind[e] != EV_READ) return bl->ev_val[e];
  return -1;
}
/* local l's value at the entry of reached block b */
static int ir_entry(Ir *ir, int b, int l) {
  int got = ir_eget(ir, b, l);
  if (got >= 0) return got;
  int n = 0, cur = b, val = -1;
  for (;;) {
    int e = ir_eget(ir, cur, l);
    if (e >= 0) { val = e; break; }
    if (cur == 0) { val = l; break; }                 /* the entry value */
    if (ir->phiblk[cur]) {
      val = ir_val_new(ir, IV_PHI, l, -1, 0, cur);
      ir->nphi++;
      if (ir->npend >= ir->cpend) {
        ir->cpend = ir->cpend ? ir->cpend * 2 : 64;
        ir->pend = realloc(ir->pend, sizeof(int) * (size_t)ir->cpend);
        if (!ir->pend) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
      }
      ir->pend[ir->npend++] = val;
      break;
    }
    ir->chain[n++] = cur;
    int p = ir->preds[ir->pred_off[cur]];
    int w = ir_last_write(ir, p, l);
    if (w >= 0) { val = w; break; }
    cur = p;
    if (n >= ir->nb) break;                            /* a cycle of single predecessors: cannot happen */
  }
  ir_eset(ir, cur, l, val);
  for (int i = 0; i < n; i++) ir_eset(ir, ir->chain[i], l, val);
  return val;
}
static void ir_fill_phis(Ir *ir) {
  while (ir->npend > 0) {
    int ph = ir->pend[--ir->npend];
    int b = ir->v[ph].block, l = ir->v[ph].local;
    int np = ir->pred_off[b + 1] - ir->pred_off[b];
    if (ir->nops + np > ir->cops) {
      while (ir->nops + np > ir->cops) ir->cops = ir->cops ? ir->cops * 2 : 256;
      ir->ops = realloc(ir->ops, sizeof(int) * (size_t)ir->cops);
      if (!ir->ops) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
    }
    int op0 = ir->nops; ir->nops += np;
    ir->v[ph].op0 = op0; ir->v[ph].nop = np;
    for (int k = 0; k < np; k++) {
      int p = ir->preds[ir->pred_off[b] + k];
      int w = ir_last_write(ir, p, l);
      ir->ops[op0 + k] = w >= 0 ? w : ir_entry(ir, p, l);
    }
  }
}
static void ir_free(Ir *ir) {
  free(ir->v); free(ir->ops); free(ir->ek); free(ir->ev); free(ir->pred_off); free(ir->preds);
  free(ir->reached); free(ir->phiblk); free(ir->rpo_pos); free(ir->chain); free(ir->pend);
  memset(ir, 0, sizeof *ir);
}

/* Builds the values of graph g, whose events name nl locals. Returns 0
   when the graph cannot take them (block 0 has a predecessor). */
static int ir_build(CfgB *g, int nl, Ir *ir) {
  memset(ir, 0, sizeof *ir);
  ir->g = g; ir->nl = nl > 0 ? nl : 1; ir->nb = g->nb;
  int nb = g->nb;
  ir->reached = ir_xalloc((size_t)nb + 1);
  ir->phiblk = ir_xalloc((size_t)nb + 1);
  ir->rpo_pos = malloc(sizeof(int) * ((size_t)nb + 1));
  ir->chain = malloc(sizeof(int) * ((size_t)nb + 1));
  int *stack = malloc(sizeof(int) * ((size_t)nb + 1)), *sidx = ir_xalloc(sizeof(int) * ((size_t)nb + 1));
  int *post = malloc(sizeof(int) * ((size_t)nb + 1));
  if (!ir->rpo_pos || !ir->chain || !stack || !post) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  for (int i = 0; i < nb; i++) ir->rpo_pos[i] = -1;
  /* reached blocks, in reverse postorder (an iterative depth-first walk) */
  int sp = 0, np = 0;
  stack[sp++] = 0; ir->reached[0] = 1;
  while (sp > 0) {
    int b = stack[sp - 1];
    if (sidx[b] < g->b[b].nsucc) {
      int s2 = g->b[b].succ[sidx[b]++];
      if (!ir->reached[s2]) { ir->reached[s2] = 1; stack[sp++] = s2; }
    } else { post[np++] = b; sp--; }
  }
  for (int i = 0; i < np; i++) ir->rpo_pos[post[np - 1 - i]] = i;
  /* the reached predecessors */
  ir->pred_off = ir_xalloc(sizeof(int) * ((size_t)nb + 2));
  for (int b = 0; b < nb; b++) if (ir->reached[b])
    for (int k = 0; k < g->b[b].nsucc; k++) ir->pred_off[g->b[b].succ[k] + 1]++;
  for (int b = 0; b < nb; b++) ir->pred_off[b + 1] += ir->pred_off[b];
  ir->preds = malloc(sizeof(int) * ((size_t)ir->pred_off[nb] + 1));
  if (!ir->preds) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  { int *fill = ir_xalloc(sizeof(int) * ((size_t)nb + 1));
    for (int b = 0; b < nb; b++) if (ir->reached[b])
      for (int k = 0; k < g->b[b].nsucc; k++) { int t = g->b[b].succ[k]; ir->preds[ir->pred_off[t] + fill[t]++] = b; }
    free(fill); }
  free(stack); free(sidx); free(post);
  if (ir->pred_off[1] - ir->pred_off[0] > 0) return 0;
  for (int b = 1; b < nb; b++) {
    if (!ir->reached[b]) continue;
    int n = ir->pred_off[b + 1] - ir->pred_off[b];
    ir->phiblk[b] = n != 1 || ir->rpo_pos[ir->preds[ir->pred_off[b]]] >= ir->rpo_pos[b];
  }
  /* the values: an entry value per local, then a value per write and per
     pi, in block order (the names a split gives follow this order) */
  for (int l = 0; l < ir->nl; l++) ir_val_new(ir, IV_ENTRY, l, -1, 0, 0);
  for (int b = 0; b < nb; b++) {
    CBlock *bl = &g->b[b];
    for (int e = 0; e < bl->nev; e++) {
      bl->ev_val[e] = -1;
      int l = bl->ev_loc[e];
      if (l < 0 || l >= ir->nl || bl->ev_kind[e] == EV_READ) continue;
      if (bl->ev_kind[e] == EV_PI) bl->ev_val[e] = ir_val_new(ir, IV_PI, l, bl->ev_node[e], bl->ev_aux[e], b);
      else bl->ev_val[e] = ir_val_new(ir, IV_DEF, l, bl->ev_node[e], bl->ev_kind[e], b);
    }
  }
  /* each read's value, and what each pi (and `&&=`) starts from */
  int *curv = malloc(sizeof(int) * (size_t)ir->nl), *touched = malloc(sizeof(int) * (size_t)ir->nl);
  if (!curv || !touched) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  for (int l = 0; l < ir->nl; l++) curv[l] = -1;
  for (int b = 0; b < nb; b++) {
    if (!ir->reached[b]) continue;
    CBlock *bl = &g->b[b];
    int nt2 = 0;
    for (int e = 0; e < bl->nev; e++) {
      int l = bl->ev_loc[e];
      if (l < 0 || l >= ir->nl) continue;
      int k = bl->ev_kind[e];
      if (k == EV_READ || k == EV_PI || k == EV_ANDWRITE) {
        int cv = curv[l] >= 0 ? curv[l] : ir_entry(ir, b, l);
        if (k == EV_READ) { bl->ev_val[e] = cv; continue; }
        ir->v[bl->ev_val[e]].src = cv;
      }
      if (curv[l] < 0) touched[nt2++] = l;
      curv[l] = bl->ev_val[e];
    }
    for (int i = 0; i < nt2; i++) curv[touched[i]] = -1;
    ir_fill_phis(ir);
  }
  free(curv); free(touched);
  /* trivial phis stand for their one value */
  for (int changed = 1; changed; ) {
    changed = 0;
    for (int x = 0; x < ir->nv; x++) {
      IrVal *ph = &ir->v[x];
      if (ph->kind != IV_PHI || ph->repl >= 0) continue;
      int same = -1, trivial = 1;
      for (int k = 0; k < ph->nop; k++) {
        int o = ir_find(ir, ir->ops[ph->op0 + k]);
        if (o == x || o == same) continue;
        if (same >= 0) { trivial = 0; break; }
        same = o;
      }
      if (trivial && same >= 0) { ir->v[x].repl = same; ir->ntrivial++; changed = 1; }
    }
  }
  /* everything names the value a trivial phi stands for */
  for (int x = 0; x < ir->nv; x++) {
    IrVal *v = &ir->v[x];
    if (v->src >= 0) v->src = ir_find(ir, v->src);
    if (v->kind == IV_PHI) for (int k = 0; k < v->nop; k++) ir->ops[v->op0 + k] = ir_find(ir, ir->ops[v->op0 + k]);
  }
  for (int b = 0; b < nb; b++) {
    CBlock *bl = &g->b[b];
    for (int e = 0; e < bl->nev; e++) if (bl->ev_kind[e] == EV_READ) bl->ev_val[e] = ir_find(ir, bl->ev_val[e]);
  }
  /* live: what a read reaches through phis, pis and `&&=` */
  { int *wl = malloc(sizeof(int) * ((size_t)ir->nv + 1)); int nw = 0;
    if (!wl) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
    for (int b = 0; b < nb; b++) {
      CBlock *bl = &g->b[b];
      for (int e = 0; e < bl->nev; e++) {
        int x = bl->ev_kind[e] == EV_READ ? bl->ev_val[e] : -1;
        if (x >= 0 && !ir->v[x].live) { ir->v[x].live = 1; wl[nw++] = x; }
      }
    }
    while (nw > 0) {
      IrVal *v = &ir->v[wl[--nw]];
      if (v->src >= 0 && !ir->v[v->src].live) { ir->v[v->src].live = 1; wl[nw++] = v->src; }
      if (v->kind == IV_PHI && v->repl < 0)
        for (int k = 0; k < v->nop; k++) {
          int o = ir->ops[v->op0 + k];
          if (o >= 0 && !ir->v[o].live) { ir->v[o].live = 1; wl[nw++] = o; }
        }
    }
    free(wl); }
  return 1;
}

/* ---- the closures' writes: a local they write keeps no fact ---- */
static void cfg_mark_captured(CfgB *g, int n, int in_closure, int depth) {
  if (n < 0 || depth > 200) return;
  const NodeTable *nt = g->nt;
  NodeKind k = nt_kind(nt, n);
  if (k == NK_DefNode || k == NK_ClassNode || k == NK_ModuleNode || k == NK_SingletonClassNode) return;
  if (k == NK_BlockNode || k == NK_LambdaNode) in_closure = 1;
  /* (full mode: a write the graph has no event for, as it is not this
     scope's, keeps no fact either) */
  if ((in_closure || (g->full && !cfg_owned(g, n))) && comp_is_local_write(k)) {
    int li = cfg_local_index(g, nt_str(nt, n, "name"));
    if (li >= 0) g->captured[li] = 1;
  }
  /* a block on the graph is no closure: its writes are events there */
  if (g->blocks && !in_closure && (k == NK_CallNode || k == NK_SuperNode || k == NK_ForwardingSuperNode)) {
    int blk = nt_ref(nt, n, "block");
    if (blk >= 0 && nt_kind(nt, blk) == NK_BlockNode && cfg_block_on_graph(g->c, n, g->full)) {
      for (int i = 0; i < nt_num_refs(nt, n); i++) {
        int ch = nt_ref_at(nt, n, i);
        if (ch != blk) { cfg_mark_captured(g, ch, 0, depth + 1); continue; }
        for (int j = 0; j < nt_num_refs(nt, blk); j++) cfg_mark_captured(g, nt_ref_at(nt, blk, j), 0, depth + 2);
        for (int j = 0; j < nt_num_arrs(nt, blk); j++) {
          int an = 0; const int *av = nt_arr_at(nt, blk, j, &an);
          for (int q = 0; q < an; q++) cfg_mark_captured(g, av[q], 0, depth + 2);
        }
      }
      for (int i = 0; i < nt_num_arrs(nt, n); i++) {
        int an = 0; const int *av = nt_arr_at(nt, n, i, &an);
        for (int q = 0; q < an; q++) cfg_mark_captured(g, av[q], 0, depth + 1);
      }
      return;
    }
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
/* The two facts on the IR: a read may run before any write when the entry
   value of a local (not a parameter's) reaches it, through phis and pis;
   and the one kind every write that reaches it stored, or BOTTOM. */
static void cfg_solve(CfgB *g) {
  int nl = g->s->nlocals;
  Ir ir; memset(&ir, 0, sizeof ir);
  if (nl <= 0 || !ir_build(g, nl, &ir)) { ir_free(&ir); return; }
  int nv = ir.nv;
  unsigned char *ty = ir_xalloc((size_t)nv + 1), *un = ir_xalloc((size_t)nv + 1);
  for (int x = 0; x < nv; x++) {
    IrVal *v = &ir.v[x];
    if (v->kind == IV_ENTRY) {
      if (g->s->locals[v->local].is_param) { ty[x] = TY_BOT; un[x] = 0; }
      else { ty[x] = (unsigned char)(TY_NIL + 1); un[x] = 1; }
    } else if (v->kind == IV_DEF && v->evk != EV_ANDWRITE) {
      ty[x] = cfg_write_kind(g, v->evk, v->node); un[x] = 0;
    } else { ty[x] = TY_TOP; un[x] = 0; }
  }
  for (int changed = 1; changed; ) {
    changed = 0;
    for (int x = 0; x < nv; x++) {
      IrVal *v = &ir.v[x];
      unsigned char t, u;
      if (v->kind == IV_PI) {
        if (v->src < 0) continue;
        t = ty[v->src]; u = un[v->src];
      } else if (v->kind == IV_DEF && v->evk == EV_ANDWRITE) {   /* `x &&= v`: unset stays */
        t = TY_BOT; u = v->src >= 0 ? un[v->src] : 0;
      } else if (v->kind == IV_PHI && v->repl < 0) {
        t = TY_TOP; u = 0;
        for (int k = 0; k < v->nop; k++) {
          int o = ir.ops[v->op0 + k];
          if (o < 0) continue;
          unsigned char ot = ty[o];
          if (t == TY_TOP) t = ot; else if (ot != TY_TOP && ot != t) t = TY_BOT;
          u |= un[o];
        }
      } else continue;
      if (t != ty[x] || u != un[x]) { ty[x] = t; un[x] = u; changed = 1; }
    }
  }
  /* each read's facts */
  for (int b = 0; b < g->nb; b++) {
    if (!ir.reached[b]) continue;
    CBlock *bl = &g->b[b];
    for (int e = 0; e < bl->nev; e++) {
      if (bl->ev_kind[e] != EV_READ) continue;
      int li = bl->ev_loc[e], node = bl->ev_node[e], x = bl->ev_val[e];
      if (li < 0 || x < 0 || g->captured[li] || node >= fc_cap) continue;
      fc_unset[node] = un[x] ? 1 : 0;
      fc_ty[node] = (ty[x] != TY_TOP && ty[x] != TY_BOT) ? (unsigned char)(ty[x] - 1) : TY_UNKNOWN;
      st_reads++;
    }
  }
  free(ty); free(un);
  ir_free(&ir);
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
  for (int i = 0; i < g.nb; i++) { free(g.b[i].ev_kind); free(g.b[i].ev_node); free(g.b[i].succ); free(g.b[i].ev_val); free(g.b[i].ev_loc); free(g.b[i].ev_aux); }
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
/* the reads the last full graph gave each pass, kept for SP_IR_DUMP */
static int ir_off(void) {
  static int v = -1;
  if (v < 0) v = getenv("SP_IR_OFF") != NULL || cfg_disabled();
  return v;
}
static int ir_log(void) {
  static int v = -1;
  if (v < 0) v = getenv("SP_IR_LOG") != NULL;
  return v;
}

/* the classic pass: the facts of the graph without pis (SP_IR_OFF) */
static int cfg_narrow_reads_classic(Compiler *c) {
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

/* ---- type narrowing on the IR -----------------------------------------

   One rule for every read of a local: the kinds of value the read can see
   (KN_*), worked out over the phis and the pis. When they are one class,
   the read unboxes to it (c->nilnarrow, codegen's read-site unbox). Two
   passes use the rule:
   - after the type fixpoint (ir_narrow_type_tests), from the pis alone:
     every write and the entry count as any value, so only the program's
     own tests prove a class. This is what the is_a? guard walk proved, on
     every shape a test can take (case/when, &&, ||, !, unless, elsif, a
     guard clause anywhere, a loop condition, a block's own tests);
   - near the end (cfg_narrow_reads), from the writes too: a write's kinds
     come from its value's type, so a read whose every write stored a
     plain Integer or Float unboxes to it. */

typedef struct { unsigned short k; int cid; } Kd;   /* cid: the one class of KN_OBJ, -2 for several */

static Kd kd_meet(Kd a, Kd b) {
  Kd r; r.k = a.k | b.k;
  if ((a.k & KN_OBJ) && (b.k & KN_OBJ)) r.cid = a.cid == b.cid ? a.cid : -2;
  else if (a.k & KN_OBJ) r.cid = a.cid;
  else if (b.k & KN_OBJ) r.cid = b.cid;
  else r.cid = -1;
  return r;
}
static Kd kd_refine(Kd a, const CfgPi *p) {
  Kd r = a;
  switch (p->op) {
  case PI_ISA: {
    unsigned short m = p->mask | (p->loose ? KN_OBJ : 0);
    r.k = a.k & m;
    if (r.k & KN_OBJ) r.cid = p->cid >= 0 && !p->loose ? p->cid : (p->cid >= 0 ? -2 : a.cid);
    break;
  }
  case PI_NOTISA:
    if (p->mask == KN_OBJ) {   /* a user class: only an object known to be of it leaves */
      if (p->cid >= 0 && !p->loose && a.cid == p->cid) r.k = a.k & ~KN_OBJ;
    } else r.k = a.k & ~p->mask;
    break;
  case PI_NIL: r.k = a.k & KN_NIL; break;
  case PI_NOTNIL: r.k = a.k & ~KN_NIL; break;
  case PI_TRUTHY: r.k = a.k & ~(KN_NIL | KN_FALSE); break;
  case PI_FALSY: r.k = a.k & (KN_NIL | KN_FALSE); break;
  }
  if (!(r.k & KN_OBJ)) r.cid = -1;
  return r;
}
/* what a write stores, from its value's type (the late pass) */
static Kd ir_def_kinds(CfgB *g, int evk, int node) {
  Kd all = { KN_ALL, -2 };
  if (evk != EV_WRITE && evk != EV_OPWRITE) return all;
  Compiler *c = g->c;
  int v = evk == EV_WRITE ? nt_ref(g->nt, node, "value") : node;
  if (v < 0) return all;
  TyKind t = infer_type(c, v);
  Kd r = { 0, -1 };
  if (t == TY_INT) r.k = cfg_narrowable(c, v, t) ? KN_INT : (KN_INT | KN_BIG | KN_NIL);
  else if (t == TY_FLOAT) r.k = cfg_narrowable(c, v, t) ? KN_FLOAT : (KN_FLOAT | KN_NIL);
  else if (t == TY_NIL) r.k = KN_NIL;
  else return all;
  return r;
}
/* The kinds of every value of the IR: entries and writes as given, then
   pis and phis to a fixpoint. `late`: the writes' own kinds, else any. */
static Kd *ir_solve_kinds(CfgB *g, Ir *ir, int late) {
  int nv = ir->nv;
  Kd *K = malloc(sizeof(Kd) * ((size_t)nv + 1));
  if (!K) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  Kd all = { KN_ALL, -2 }, nil = { KN_NIL, -1 }, none = { 0, -1 };
  for (int x = 0; x < nv; x++) {
    IrVal *v = &ir->v[x];
    if (v->kind == IV_ENTRY) K[x] = late && !g->s->locals[v->local].is_param ? nil : all;
    else if (v->kind == IV_DEF) K[x] = late ? ir_def_kinds(g, v->evk, v->node) : all;
    else K[x] = none;
  }
  for (int changed = 1, round = 0; changed && round < 1000; round++) {
    changed = 0;
    for (int x = 0; x < nv; x++) {
      IrVal *v = &ir->v[x];
      Kd r;
      if (v->kind == IV_PI) {
        if (v->src < 0) continue;
        r = kd_refine(K[v->src], &g->pis[v->evk]);
      } else if (v->kind == IV_PHI && v->repl < 0) {
        r = none;
        for (int q = 0; q < v->nop; q++) { int o = ir->ops[v->op0 + q]; if (o >= 0) r = kd_meet(r, K[o]); }
      } else continue;
      if (r.k != K[x].k || r.cid != K[x].cid) { K[x] = r; changed = 1; }
    }
  }
  return K;
}
/* the full graph of scope s and its IR; 0 when it has none */
static int ir_scope_build(Compiler *c, Scope *s, CfgB *g, Ir *ir, int late) {
  memset(g, 0, sizeof *g); memset(ir, 0, sizeof *ir);
  if (s->body < 0 || s->cs_synth || s->nlocals <= 0) return 0;
  g->c = c; g->nt = c->nt; g->s = s; g->si = s->idx; g->want_types = late;
  g->full = 1; g->blocks = 1;
  g->captured = calloc((size_t)s->nlocals + 1, 1);
  if (!g->captured) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  for (int i = 0; i < s->nlocals; i++)
    if (s->locals[i].proc_rebinds || (late && s->locals[i].is_cell)) g->captured[i] = 1;
  cfg_mark_captured(g, s->body, 0, 0);
  g->cur = cfg_new_block(g);
  g->exit_blk = cfg_new_block(g);
  cfg_lower(g, s->body);
  cfg_jump_to(g, g->exit_blk);
  return ir_build(g, s->nlocals, ir);
}
static void ir_scope_free(CfgB *g, Ir *ir) {
  for (int i = 0; i < g->nb; i++) {
    CBlock *bl = &g->b[i];
    free(bl->ev_kind); free(bl->ev_node); free(bl->succ); free(bl->ev_val); free(bl->ev_loc); free(bl->ev_aux);
  }
  free(g->b); free(g->captured); free(g->pis);
  ir_free(ir);
}
/* the class a read's kinds prove, by name, or NULL */
static const char *kd_class(Compiler *c, Kd k) {
  if (k.k == KN_STR) return "String";
  if (k.k == KN_SYM) return "Symbol";
  if (k.k == KN_FLOAT) return "Float";
  if (k.k && !(k.k & ~(KN_INT | KN_BIG))) return "Integer";
  if (k.k == KN_ARR) return "Array";
  if (k.k == KN_OBJ && k.cid >= 0 && k.cid < c->nclasses) return c->classes[k.cid].name;
  return NULL;
}
static void kd_print(Compiler *c, Kd k, FILE *f) {
  static const char *const nm[] = { "nil", "true", "false", "Integer", "Bignum", "Float", "String",
    "Symbol", "Array", "Hash", "Object", "other" };
  if (k.k == KN_ALL || (k.k & KN_UNK)) { fprintf(f, "any"); return; }
  if (!k.k) { fprintf(f, "none"); return; }
  int any = 0;
  for (int b = 0; b < 12; b++) if (k.k & (1 << b)) {
    if (b == 10 && k.cid >= 0 && k.cid < c->nclasses) fprintf(f, "%s%s", any ? "|" : "", c->classes[k.cid].name);
    else fprintf(f, "%s%s", any ? "|" : "", nm[b]);
    any = 1;
  }
}
static const char *pi_op_name(int op) {
  switch (op) {
  case PI_ISA: return "is"; case PI_NOTISA: return "is not"; case PI_NIL: return "is nil";
  case PI_NOTNIL: return "is not nil"; case PI_TRUTHY: return "truthy"; case PI_FALSY: return "falsy";
  }
  return "?";
}
/* the name of local li of graph g (an SSA graph has its own list: the
   scope's locals are registered after the rename) */
static const char *ir_local_name(CfgB *g, int li) {
  if (g->ssa) return li >= 0 && li < g->nsn ? g->sn[li] : "?";
  return li >= 0 && li < g->s->nlocals ? g->s->locals[li].name : "?";
}
/* the values of what write x stores, before its Write line */
static void ir_dump_owned(Compiler *c, Ir *ir, Kd *K, int x, FILE *f) {
  const NodeTable *nt = c->nt;
  for (int y = 0; y < ir->nv; y++) {
    IrVal *v = &ir->v[y];
    if (v->owner != x || v->kind == IV_DEF) continue;
    fprintf(f, "  v%d = ", y);
    int n = v->node;
    NodeKind nk = n >= 0 ? nt_kind(nt, n) : NK_NilNode;
    switch (v->kind) {
    case IV_CONST:
      if (n < 0 || nk == NK_NilNode) fprintf(f, "Const nil");
      else if (nk == NK_IntegerNode && nt_str(nt, n, "bigval")) fprintf(f, "Const %s", nt_str(nt, n, "bigval"));
      else if (nk == NK_IntegerNode) fprintf(f, "Const %lld", nt_int(nt, n, "value", 0));
      else if (nk == NK_TrueNode) fprintf(f, "Const true");
      else if (nk == NK_FalseNode) fprintf(f, "Const false");
      else if (nk == NK_SymbolNode && nt_str(nt, n, "value")) fprintf(f, "Const :%s", nt_str(nt, n, "value"));
      else if (nk == NK_StringNode && nt_str(nt, n, "content")) fprintf(f, "Const \"%.20s\"", nt_str(nt, n, "content"));
      else if (nk == NK_FloatNode && nt_content(nt, n)) fprintf(f, "Const %s", nt_content(nt, n));
      else fprintf(f, "Const %s", nt_type(nt, n) ? nt_type(nt, n) : "?");
      break;
    case IV_SEND: {
      const char *m = nk == NK_CallNode ? nt_str(nt, n, "name") : nt_str(nt, n, "binary_operator");
      int rv = v->nop > 0 ? ir->ops[v->op0] : -1;
      if (rv >= 0) fprintf(f, "Send v%d, :%s", rv, m ? m : "?");
      else fprintf(f, "Send self, :%s", m ? m : "?");
      for (int q = 1; q < v->nop; q++) fprintf(f, ", v%d", ir->ops[v->op0 + q]);
      if (v->evk & SF_BLOCK) fprintf(f, ", block");
      break;
    }
    case IV_JOIN: case IV_AND: case IV_OR:
      fprintf(f, "%s", v->kind == IV_JOIN ? "Phi" : v->kind == IV_AND ? "And" : "Or");
      for (int q = 0; q < v->nop; q++) fprintf(f, "%s v%d", q ? "," : "", ir->ops[v->op0 + q]);
      break;
    case IV_RESCUED: fprintf(f, "Rescued"); break;
    default: fprintf(f, "Other %s", n >= 0 && nt_type(nt, n) ? nt_type(nt, n) : "?"); break;
    }
    fprintf(f, " : "); kd_print(c, K[y], f); fprintf(f, "\n");
  }
}
/* SP_IR_DUMP=name prints the IR of each method of that name, in the shape
   ZJIT's HIR dump has: blocks, their phis, and one line per value */
static void ir_dump(Compiler *c, Scope *s, CfgB *g, Ir *ir, Kd *K, const char *pass) {
  const char *want = getenv("SP_IR_DUMP");
  if (!want || !s->name || !sp_streq(want, s->name)) return;
  FILE *f = stderr;
  fprintf(f, "fn %s (%s pass):\n", s->name, pass);
  int *order = malloc(sizeof(int) * ((size_t)g->nb + 1)), no = 0;
  if (!order) return;
  for (int b = 0; b < g->nb; b++) if (ir->reached[b]) order[no++] = b;
  for (int i = 1; i < no; i++) for (int j = i; j > 0 && ir->rpo_pos[order[j]] < ir->rpo_pos[order[j - 1]]; j--) {
    int t = order[j]; order[j] = order[j - 1]; order[j - 1] = t;
  }
  for (int i = 0; i < no; i++) {
    int b = order[i];
    CBlock *bl = &g->b[b];
    fprintf(f, "bb%d:", b);
    int np = ir->pred_off[b + 1] - ir->pred_off[b];
    if (np) { fprintf(f, "  (from"); for (int q = 0; q < np; q++) fprintf(f, " bb%d", ir->preds[ir->pred_off[b] + q]); fprintf(f, ")"); }
    fprintf(f, "\n");
    if (b == 0)
      for (int x = 0; x < ir->nv; x++) if (ir->v[x].kind == IV_ENTRY && ir->v[x].live) {
        fprintf(f, "  v%d = Entry %s : ", x, ir_local_name(g, ir->v[x].local)); kd_print(c, K[x], f); fprintf(f, "\n");
      }
    for (int x = 0; x < ir->nv; x++) {
      IrVal *v = &ir->v[x];
      if (v->kind != IV_PHI || v->block != b || v->repl >= 0 || !v->live) continue;
      fprintf(f, "  v%d = Phi %s", x, ir_local_name(g, v->local));
      for (int q = 0; q < v->nop; q++) fprintf(f, "%s v%d", q ? "," : "", ir->ops[v->op0 + q]);
      fprintf(f, " : "); kd_print(c, K[x], f); fprintf(f, "\n");
    }
    for (int e = 0; e < bl->nev; e++) {
      int li = bl->ev_loc[e], x = bl->ev_val[e], node = bl->ev_node[e];
      if (li < 0 || x < 0) continue;
      int line = node >= 0 ? (int)nt_int(c->nt, node, "node_line", 0) : 0;
      if (bl->ev_kind[e] == EV_READ) {
        fprintf(f, "  Read %s = v%d : ", ir_local_name(g, li), x); kd_print(c, K[x], f);
        if (!g->ssa && node >= 0 && c->nilnarrow && c->nilnarrow[node] != TY_UNKNOWN) {
          TyKind nt2 = c->nilnarrow[node];
          int oc = ty_is_object(nt2) ? ty_object_class(nt2) : -1;
          fprintf(f, "  -> unboxed as %s", oc >= 0 && oc < c->nclasses ? c->classes[oc].name : ty_name(nt2));
        }
        fprintf(f, "   (line %d)\n", line);
      } else if (bl->ev_kind[e] == EV_PI) {
        const CfgPi *p = &g->pis[bl->ev_aux[e]];
        fprintf(f, "  v%d = Pi v%d, %s %s %s : ", x, ir->v[x].src, ir_local_name(g, li), pi_op_name(p->op), p->cls ? p->cls : "");
        kd_print(c, K[x], f); fprintf(f, "   (line %d)\n", line);
      } else {
        ir_dump_owned(c, ir, K, x, f);
        fprintf(f, "  v%d = Write %s", x, ir_local_name(g, li));
        if (ir->v[x].arg >= 0) fprintf(f, " v%d", ir->v[x].arg);
        fprintf(f, " : "); kd_print(c, K[x], f); fprintf(f, "   (line %d)\n", line);
      }
    }
    int ns = bl->nsucc;
    if (ns) { fprintf(f, "  -> "); for (int q = 0; q < ns; q++) fprintf(f, "%sbb%d", q ? ", " : "", bl->succ[q]); fprintf(f, "\n"); }
  }
  free(order);
}

/* ---- the IR's own types: a value for each step of what a write stores --

   A Write above stores a value the IR did not see: the value subtree of
   its node. ir_add_exprs gives each step of that subtree a value of its
   own, as ZJIT's HIR does:

     y = a + 1        v7 = Const 1
                      v8 = Send v4, :+, v7      (v4: the value `a` has there)
                      v9 = Write y v8

   A literal is a Const, a call a Send (its operands: the receiver, -1 for
   a call on self, then the arguments), the arms of an `if`, a `case` or a
   `rescue` modifier meet at a Phi, `a && b` is an And and `a || b` an Or,
   the exception a rescue arm binds is a Rescued, and anything else is an
   Other. A local read makes no value: it names the value its event sees.
   Each value keeps the write it belongs to (owner), so the dump prints it
   just before that write.

   ir_own_kinds then works out the kinds of every value from these alone,
   with no type from Spinel's inference: a Const has its literal's kind, a
   Send the kind a rule for its method gives on its receiver's and its
   arguments' kinds (an Integer plus an Integer is an Integer), a Write the
   kinds of what it stores, and a Phi the kinds of all its operands. A
   parameter's entry, an Other and a Send with no rule are any value
   (KN_UNK). */


typedef struct {
  Ir *ir; const NodeTable *nt;
  const int *rd, *wd; int cap;    /* per node: the value a read sees, the DEF a write makes */
  const int *resc; int nresc;     /* the targets rescue arms bind */
  int owner, block, depth;
  char *built;                    /* per DEF (below nbase): its value is made */
  int nbase;
} IrX;

static int irx_new(IrX *x, int kind, int node, const int *ops, int nop) {
  Ir *ir = x->ir;
  int v = ir_val_new(ir, kind, -1, node, 0, x->block);
  ir->v[v].owner = x->owner;
  if (nop > 0) {
    if (ir->nops + nop > ir->cops) {
      while (ir->nops + nop > ir->cops) ir->cops = ir->cops ? ir->cops * 2 : 256;
      ir->ops = realloc(ir->ops, sizeof(int) * (size_t)ir->cops);
      if (!ir->ops) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
    }
    ir->v[v].op0 = ir->nops; ir->v[v].nop = nop;
    memcpy(ir->ops + ir->nops, ops, sizeof(int) * (size_t)nop);
    ir->nops += nop;
  }
  return v;
}
static int irx_value(IrX *x, int n);
static void irx_def(IrX *x, int d);
/* the value of a body: its last statement's, nil when it is empty */
static int irx_body(IrX *x, int n) {
  if (n < 0) return irx_new(x, IV_CONST, -1, NULL, 0);
  if (nt_kind(x->nt, n) != NK_StatementsNode) return irx_value(x, n);
  int bn = 0; const int *b = nt_arr(x->nt, n, "body", &bn);
  return bn > 0 ? irx_value(x, b[bn - 1]) : irx_new(x, IV_CONST, -1, NULL, 0);
}
static int irx_value(IrX *x, int n) {
  const NodeTable *nt = x->nt;
  if (n < 0) return irx_new(x, IV_CONST, -1, NULL, 0);
  if (x->depth > 200) return irx_new(x, IV_OTHER, n, NULL, 0);
  x->depth++;
  int r, ops[40], no = 0;
  NodeKind k = nt_kind(nt, n);
  switch (k) {
  case NK_IntegerNode: case NK_FloatNode: case NK_RationalNode: case NK_ImaginaryNode:
  case NK_StringNode: case NK_InterpolatedStringNode: case NK_XStringNode: case NK_InterpolatedXStringNode:
  case NK_SymbolNode: case NK_InterpolatedSymbolNode: case NK_NilNode: case NK_TrueNode: case NK_FalseNode:
  case NK_ArrayNode: case NK_HashNode: case NK_RangeNode: case NK_RegularExpressionNode:
  case NK_InterpolatedRegularExpressionNode: case NK_LambdaNode: case NK_SourceFileNode:
  case NK_SourceLineNode: case NK_DefinedNode:
    r = irx_new(x, IV_CONST, n, NULL, 0);
    break;
  case NK_LocalVariableReadNode:
    r = n < x->cap && x->rd[n] >= 0 ? x->rd[n] : irx_new(x, IV_OTHER, n, NULL, 0);
    break;
  case NK_LocalVariableWriteNode: case NK_LocalVariableOperatorWriteNode:
  case NK_LocalVariableOrWriteNode: case NK_LocalVariableAndWriteNode: {
    /* `x = (y = v)`: x stores what y does */
    int d = n < x->cap ? x->wd[n] : -1;
    if (d >= 0 && d < x->nbase && x->ir->v[d].kind == IV_DEF) { irx_def(x, d); r = d; }
    else r = irx_new(x, IV_OTHER, n, NULL, 0);
    break;
  }
  case NK_InstanceVariableWriteNode: case NK_GlobalVariableWriteNode: case NK_ClassVariableWriteNode:
  case NK_ConstantWriteNode:
    r = irx_value(x, nt_ref(nt, n, "value"));
    break;
  case NK_ParenthesesNode: r = irx_body(x, nt_ref(nt, n, "body")); break;
  case NK_StatementsNode: r = irx_body(x, n); break;
  case NK_BeginNode:
    if (nt_ref(nt, n, "rescue_clause") < 0 && nt_ref(nt, n, "else_clause") < 0) r = irx_body(x, nt_ref(nt, n, "statements"));
    else r = irx_new(x, IV_OTHER, n, NULL, 0);
    break;
  case NK_IfNode: case NK_UnlessNode: {
    int els = nt_ref(nt, n, k == NK_IfNode ? "subsequent" : "else_clause");
    ops[no++] = irx_body(x, nt_ref(nt, n, "statements"));
    ops[no++] = els >= 0 ? irx_value(x, els) : irx_new(x, IV_CONST, -1, NULL, 0);
    r = irx_new(x, IV_JOIN, n, ops, no);
    break;
  }
  case NK_ElseNode: r = irx_body(x, nt_ref(nt, n, "statements")); break;
  case NK_CaseNode: case NK_CaseMatchNode: {
    int wn = 0; const int *w = nt_arr(nt, n, "conditions", &wn);
    if (wn > 32) { r = irx_new(x, IV_OTHER, n, NULL, 0); break; }
    for (int i = 0; i < wn; i++) ops[no++] = irx_body(x, nt_ref(nt, w[i], "statements"));
    int els = nt_ref(nt, n, "else_clause");
    if (els >= 0) ops[no++] = irx_value(x, els);
    /* no arm matched: a `case` gives nil, a `case ... in` raises */
    else if (k == NK_CaseNode) ops[no++] = irx_new(x, IV_CONST, -1, NULL, 0);
    r = irx_new(x, IV_JOIN, n, ops, no);
    break;
  }
  case NK_AndNode: case NK_OrNode:
    ops[no++] = irx_value(x, nt_ref(nt, n, "left"));
    ops[no++] = irx_value(x, nt_ref(nt, n, "right"));
    r = irx_new(x, k == NK_AndNode ? IV_AND : IV_OR, n, ops, no);
    break;
  case NK_RescueModifierNode:
    ops[no++] = irx_value(x, nt_ref(nt, n, "expression"));
    ops[no++] = irx_value(x, nt_ref(nt, n, "rescue_expression"));
    r = irx_new(x, IV_JOIN, n, ops, no);
    break;
  case NK_CallNode: {
    int rc = nt_ref(nt, n, "receiver"), args = nt_ref(nt, n, "arguments"), blk = nt_ref(nt, n, "block");
    int an = 0; const int *av = args >= 0 ? nt_arr(nt, args, "arguments", &an) : NULL;
    if (an > 32) { r = irx_new(x, IV_OTHER, n, NULL, 0); break; }
    int fl = (blk >= 0 ? SF_BLOCK : 0) | (nt_safe_nav(nt, n) ? SF_SAFE : 0);
    ops[no++] = rc >= 0 ? irx_value(x, rc) : -1;
    for (int i = 0; i < an; i++) {
      NodeKind ak = nt_kind(nt, av[i]);
      if (ak == NK_SplatNode || ak == NK_KeywordHashNode || ak == NK_BlockArgumentNode ||
          ak == NK_ForwardingArgumentsNode) fl |= SF_ODD;
      ops[no++] = irx_value(x, av[i]);
    }
    r = irx_new(x, IV_SEND, n, ops, no);
    x->ir->v[r].evk = fl;
    break;
  }
  default:
    r = irx_new(x, IV_OTHER, n, NULL, 0);
    break;
  }
  x->depth--;
  return r;
}
/* what DEF d stores */
static void irx_def(IrX *x, int d) {
  if (d < 0 || d >= x->nbase || x->built[d]) return;
  x->built[d] = 1;
  Ir *ir = x->ir;
  const NodeTable *nt = x->nt;
  int node = ir->v[d].node, evk = ir->v[d].evk;
  if (node < 0) return;
  int so = x->owner, sb = x->block;
  x->owner = d; x->block = ir->v[d].block;
  NodeKind nk = nt_kind(nt, node);
  int a = -1, ops[2];
  if (evk == EV_WRITE) a = irx_value(x, nt_ref(nt, node, "value"));
  else if (evk == EV_OPWRITE) {          /* `x += v`: x's value before, then v */
    ops[0] = node < x->cap ? x->rd[node] : -1;
    ops[1] = irx_value(x, nt_ref(nt, node, "value"));
    a = irx_new(x, IV_SEND, node, ops, 2);
  } else if (evk == EV_ANDWRITE) {       /* `x &&= v` */
    ops[0] = ir->v[d].src;
    ops[1] = irx_value(x, nt_ref(nt, node, "value"));
    a = irx_new(x, IV_AND, node, ops, 2);
  } else if (evk == EV_WRITE_UNK && nk == NK_LocalVariableOrWriteNode) {   /* `x ||= v` */
    ops[0] = node < x->cap ? x->rd[node] : -1;
    ops[1] = irx_value(x, nt_ref(nt, node, "value"));
    a = irx_new(x, IV_OR, node, ops, 2);
  } else if (evk == EV_WRITE_UNK && nk == NK_LocalVariableTargetNode) {
    for (int i = 0; i < x->nresc; i++) if (x->resc[i] == node) { a = irx_new(x, IV_RESCUED, node, NULL, 0); break; }
  }
  ir->v[d].arg = a;
  x->owner = so; x->block = sb;
}
/* the values of what each write of a reached block stores; rd and wd map
   a node to the value its read sees and to the DEF its write makes */
static void ir_add_exprs(Ir *ir, const NodeTable *nt, const int *rd, const int *wd, int cap,
                         const int *resc, int nresc) {
  IrX x; memset(&x, 0, sizeof x);
  x.ir = ir; x.nt = nt; x.rd = rd; x.wd = wd; x.cap = cap; x.resc = resc; x.nresc = nresc;
  x.owner = -1; x.nbase = ir->nv;
  x.built = ir_xalloc((size_t)ir->nv + 1);
  for (int d = 0; d < x.nbase; d++)
    if (ir->v[d].kind == IV_DEF && ir->reached[ir->v[d].block]) irx_def(&x, d);
  free(x.built);
}

/* The method names the program defines (its prelude aside): any of them
   (a call on self can reach one), and those on the class of a builtin kind
   (a reopen of Integer or String can change what its `+` gives). A rule
   for a name the program defines does not apply. */
typedef struct { char **all; int nall; char **core; int ncore; } IrNames;
static IrNames g_ir_names;
static int ir_names_cmp(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }
static int ir_names_has(char **l, int n, const char *m) {
  return n > 0 && m && bsearch(&m, l, (size_t)n, sizeof(char *), ir_names_cmp) != NULL;
}
static void ir_names_build(Compiler *c) {
  IrNames *R = &g_ir_names;
  R->all = malloc(sizeof(char *) * ((size_t)c->nscopes + 1));
  R->core = malloc(sizeof(char *) * ((size_t)c->nscopes + 1));
  if (!R->all || !R->core) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  R->nall = R->ncore = 0;
  for (int si = 0; si < c->nscopes; si++) {
    Scope *s = &c->scopes[si];
    if (s->def_node < 0 || !s->name || nt_kind(c->nt, s->def_node) != NK_DefNode) continue;
    if (nt_int(c->nt, s->def_node, "node_bi", 0)) continue;
    R->all[R->nall++] = s->name;
    if (s->class_id >= 0 && s->class_id < c->nclasses && is_builtin_reopen(c->classes[s->class_id].name))
      R->core[R->ncore++] = s->name;
  }
  qsort(R->all, (size_t)R->nall, sizeof(char *), ir_names_cmp);
  qsort(R->core, (size_t)R->ncore, sizeof(char *), ir_names_cmp);
}
static void ir_names_free(void) {
  free(g_ir_names.all); free(g_ir_names.core);
  memset(&g_ir_names, 0, sizeof g_ir_names);
}

static int ir_m_in(const char *m, const char *const *l) {
  for (; *l; l++) if (sp_streq(m, *l)) return 1;
  return 0;
}
static const char *const RM_TEST[] = { "==", "!=", "<", ">", "<=", ">=", "===", "eql?", "equal?",
  "between?", "nil?", "zero?", "positive?", "negative?", "even?", "odd?", "integer?", "finite?", "nan?",
  "is_a?", "kind_of?", "instance_of?", "frozen?", "empty?", "include?", "start_with?", "end_with?",
  "key?", "has_key?", "member?", "value?", "has_value?", "any?", "all?", "none?", "ascii_only?",
  "match?", "!", NULL };
static const char *const RM_SAME[] = { "dup", "clone", "itself", "freeze", NULL };
static const char *const RM_NUM_OP[] = { "+", "-", "*", "/", "%", "modulo", NULL };
static const char *const RM_INT_BITS[] = { "&", "|", "^", "<<", ">>", "gcd", "lcm", NULL };
static const char *const RM_INT_INT[] = { "-@", "+@", "~", "abs", "magnitude", "succ", "next", "pred",
  "ord", "floor", "ceil", "round", "truncate", "to_i", "to_int", "bit_length", "size", "hash", NULL };
static const char *const RM_FLT_INT[] = { "floor", "ceil", "round", "truncate", "to_i", "to_int", "hash", NULL };
static const char *const RM_STR_STR[] = { "upcase", "downcase", "capitalize", "swapcase", "strip",
  "lstrip", "rstrip", "chomp", "chop", "chr", "reverse", "squeeze", "succ", "next", "to_s", "to_str",
  "inspect", "dump", "center", "ljust", "rjust", "tr", "tr_s", "delete", "sub", "encode",
  "force_encoding", "unicode_normalize", "scrub", "b", "+@", "-@", "dedup", "%", "<<", "concat",
  "prepend", "insert", "replace", NULL };
static const char *const RM_STR_BANG[] = { "upcase!", "downcase!", "capitalize!", "swapcase!", "strip!",
  "lstrip!", "rstrip!", "chomp!", "chop!", "squeeze!", "sub!", "gsub!", "tr!", "delete!", "[]",
  "slice", "byteslice", NULL };
static const char *const RM_STR_INT[] = { "size", "length", "bytesize", "count", "ord", "hex", "oct",
  "to_i", "hash", "sum", NULL };
static const char *const RM_STR_ARR[] = { "split", "chars", "bytes", "lines", "codepoints", "scan",
  "partition", "rpartition", "unpack", NULL };
static const char *const RM_NIL_RAISE[] = { "+", "-", "*", "/", "%", "**", "<", ">", "<=", ">=",
  "<<", ">>", "[]", "size", "length", NULL };
static const char *const RM_ARR_ARR[] = { "to_a", "entries", "compact", "flatten", "uniq", "sort",
  "reverse", "rotate", "shuffle", "push", "append", "<<", "concat", "unshift", "prepend", "insert",
  "fill", "clear", "replace", "drop", "take", "zip", "product", "transpose", "values_at", NULL };

/* the kinds the numeric operator m gives on an Integer (or a Float,
   flt) receiver: per kind of its argument, an Integer, a Float, a raise
   (TypeError), or any value (an object can coerce) */
static unsigned short ir_num_op(int flt, int bits, unsigned short a) {
  unsigned short r = 0;
  if (a & KN_INTS) r |= flt ? KN_FLOAT : (KN_INT | (a & KN_BIG));
  if (a & KN_FLOAT) r |= bits && !flt ? 0 : KN_FLOAT;
  if (a & (KN_OBJ | KN_OTHER | KN_UNK)) r |= KN_ALL | KN_UNK;
  return r;
}
/* what method m gives on a receiver of the one kind bit, with these
   arguments; KN_UNK when no rule says */
static unsigned short ir_rule(int bit, const char *m, int na, const Kd *ak) {
  const unsigned short any = KN_ALL | KN_UNK;
  unsigned short a0 = na > 0 ? ak[0].k : 0;
  if (sp_streq(m, "to_s") || sp_streq(m, "inspect")) return bit & (KN_OBJ | KN_OTHER) ? any : KN_STR;
  if (ir_m_in(m, RM_SAME)) return bit & (KN_OBJ | KN_OTHER) ? any : (unsigned short)bit;
  if (bit & (KN_OBJ | KN_OTHER)) return any;
  if (ir_m_in(m, RM_TEST)) return KN_BOOL;
  if (sp_streq(m, "<=>")) return KN_INT | KN_NIL;
  switch (bit) {
  case KN_INT: case KN_BIG:
    if (na == 1 && (ir_m_in(m, RM_NUM_OP) || ir_m_in(m, RM_INT_BITS))) {
      unsigned short k = ir_num_op(0, !ir_m_in(m, RM_NUM_OP), a0);
      return k & KN_INT ? (unsigned short)(k | (bit & KN_BIG)) : k;
    }
    if (na == 0 && ir_m_in(m, RM_INT_INT)) return KN_INT | (bit & KN_BIG);
    if ((na == 0 && sp_streq(m, "to_f")) || (na == 1 && sp_streq(m, "fdiv"))) return KN_FLOAT;
    if (na == 0 && sp_streq(m, "chr")) return KN_STR;
    if (na == 0 && (sp_streq(m, "digits") || sp_streq(m, "divmod"))) return KN_ARR;
    if (na == 1 && sp_streq(m, "divmod")) return KN_ARR;
    if (na == 0 && (sp_streq(m, "to_r") || sp_streq(m, "rationalize") || sp_streq(m, "to_c"))) return KN_OTHER;
    return any;
  case KN_FLOAT:
    if (na == 1 && (ir_m_in(m, RM_NUM_OP) || sp_streq(m, "quo") || sp_streq(m, "fdiv"))) return ir_num_op(1, 0, a0);
    if (na == 0 && (sp_streq(m, "-@") || sp_streq(m, "+@") || sp_streq(m, "abs") || sp_streq(m, "magnitude") ||
                    sp_streq(m, "to_f"))) return KN_FLOAT;
    if (na == 0 && ir_m_in(m, RM_FLT_INT)) return KN_INT;
    if (na == 1 && sp_streq(m, "divmod")) return KN_ARR;
    if (na == 0 && (sp_streq(m, "to_r") || sp_streq(m, "rationalize"))) return KN_OTHER;
    return any;
  case KN_STR:
    if (na == 1 && sp_streq(m, "+"))
      return (a0 & KN_STR ? KN_STR : 0) | (a0 & (KN_OBJ | KN_OTHER | KN_UNK) ? any : 0);
    if (na == 1 && sp_streq(m, "*"))
      return (a0 & (KN_INTS | KN_FLOAT) ? KN_STR : 0) | (a0 & (KN_OBJ | KN_OTHER | KN_UNK) ? any : 0);
    if (ir_m_in(m, RM_STR_STR)) return KN_STR;
    if (sp_streq(m, "gsub")) return na >= 2 ? KN_STR : na == 1 ? KN_OTHER : any;
    if (ir_m_in(m, RM_STR_BANG)) return KN_STR | KN_NIL;
    if (ir_m_in(m, RM_STR_INT)) return KN_INT;
    if (sp_streq(m, "index") || sp_streq(m, "rindex") || sp_streq(m, "byteindex") || sp_streq(m, "=~") ||
        sp_streq(m, "casecmp")) return KN_INT | KN_NIL;
    if (na == 0 && sp_streq(m, "to_f")) return KN_FLOAT;
    if (na == 0 && (sp_streq(m, "to_sym") || sp_streq(m, "intern"))) return KN_SYM;
    if (na == 0 && (sp_streq(m, "to_r") || sp_streq(m, "to_c"))) return KN_OTHER;
    if (ir_m_in(m, RM_STR_ARR)) return KN_ARR;
    return any;
  case KN_SYM:
    if (na == 0 && (sp_streq(m, "id2name") || sp_streq(m, "name"))) return KN_STR;
    if (na == 0 && (sp_streq(m, "to_sym") || sp_streq(m, "upcase") || sp_streq(m, "downcase") ||
                    sp_streq(m, "capitalize") || sp_streq(m, "swapcase") || sp_streq(m, "succ") ||
                    sp_streq(m, "next"))) return KN_SYM;
    if (na == 0 && (sp_streq(m, "size") || sp_streq(m, "length"))) return KN_INT;
    if (sp_streq(m, "[]")) return KN_STR | KN_NIL;
    if (na == 0 && sp_streq(m, "to_proc")) return KN_OTHER;
    return any;
  case KN_NIL: case KN_TRUE: case KN_FALSE:
    if (ir_m_in(m, RM_NIL_RAISE)) return 0;   /* NoMethodError */
    if (sp_streq(m, "&") || sp_streq(m, "|") || sp_streq(m, "^")) return KN_BOOL;
    if (bit != KN_NIL || na != 0) return any;
    if (sp_streq(m, "to_a")) return KN_ARR;
    if (sp_streq(m, "to_h")) return KN_HASH;
    if (sp_streq(m, "to_i")) return KN_INT;
    if (sp_streq(m, "to_f")) return KN_FLOAT;
    if (sp_streq(m, "to_r")) return KN_OTHER;
    return any;
  case KN_ARR:
    if (na == 0 && (sp_streq(m, "size") || sp_streq(m, "length") || sp_streq(m, "count"))) return KN_INT;
    if (sp_streq(m, "index") || sp_streq(m, "find_index") || sp_streq(m, "rindex")) return KN_INT | KN_NIL;
    if (sp_streq(m, "join") || sp_streq(m, "pack")) return KN_STR;
    if (na == 1 && sp_streq(m, "*"))
      return (a0 & KN_STR ? KN_STR : 0) | (a0 & (KN_INTS | KN_FLOAT) ? KN_ARR : 0) |
             (a0 & (KN_OBJ | KN_OTHER | KN_UNK) ? any : 0);
    if (na == 1 && (sp_streq(m, "+") || sp_streq(m, "-") || sp_streq(m, "&") || sp_streq(m, "|")))
      return (a0 & KN_ARR ? KN_ARR : 0) | (a0 & (KN_OBJ | KN_OTHER | KN_UNK) ? any : 0);
    if (ir_m_in(m, RM_ARR_ARR)) return KN_ARR;
    if (na == 1 && (sp_streq(m, "first") || sp_streq(m, "last") || sp_streq(m, "sample") ||
                    sp_streq(m, "min") || sp_streq(m, "max"))) return KN_ARR;
    if (na == 0 && (sp_streq(m, "to_h") || sp_streq(m, "tally"))) return KN_HASH;
    return any;
  case KN_HASH:
    if (na == 0 && (sp_streq(m, "size") || sp_streq(m, "length") || sp_streq(m, "count"))) return KN_INT;
    if (na == 0 && (sp_streq(m, "keys") || sp_streq(m, "values") || sp_streq(m, "to_a"))) return KN_ARR;
    if ((na == 0 && (sp_streq(m, "to_h") || sp_streq(m, "invert") || sp_streq(m, "compact"))) ||
        (na == 1 && sp_streq(m, "merge"))) return KN_HASH;
    return any;
  }
  return any;
}
/* the kinds `K.new` (and a few calls on a class) give, by the class's
   name: a class of the program's whose superclasses are all its own makes
   an object of that class */
static Kd ir_class_call_kinds(Compiler *c, const char *cn, const char *m) {
  Kd any = { KN_ALL | KN_UNK, -2 }, r = { 0, -1 };
  if (!cn || !m) return any;
  if (sp_streq(m, "new")) {
    if (sp_streq(cn, "Array")) { r.k = KN_ARR; return r; }
    if (sp_streq(cn, "Hash")) { r.k = KN_HASH; return r; }
    if (sp_streq(cn, "String")) { r.k = KN_STR; return r; }
    if (is_builtin_exception_name(cn) || sp_streq(cn, "Struct") || sp_streq(cn, "Proc") ||
        sp_streq(cn, "Range") || sp_streq(cn, "Set") || sp_streq(cn, "Time")) { r.k = KN_OTHER; return r; }
    int ci = comp_class_index(c, cn);
    if (ci < 0 || comp_cmethod_in_chain(c, ci, "new", NULL) >= 0) return any;   /* a `def self.new` */
    for (int k = ci, d = 0; k >= 0 && d < 64; d++) {
      if (k >= c->nclasses || is_builtin_reopen(c->classes[k].name) || c->classes[k].is_struct) return any;
      int dn = c->classes[k].def_node;
      if (dn < 0 || nt_kind(c->nt, dn) != NK_ClassNode) return any;
      int sup = nt_ref(c->nt, dn, "superclass");
      if (sup < 0) { r.k = KN_OBJ; r.cid = ci; return r; }
      if (nt_kind(c->nt, sup) != NK_ConstantReadNode) return any;
      k = comp_class_index(c, nt_str(c->nt, sup, "name"));
    }
    return any;
  }
  return any;
}
/* the kinds a receiver-less call gives: Kernel's conversions and a few
   others, when the program defines no method of that name */
static unsigned short ir_self_rule(const char *m, int na) {
  const unsigned short any = KN_ALL | KN_UNK;
  if (ir_names_has(g_ir_names.all, g_ir_names.nall, m)) return any;
  if (na == 1 && sp_streq(m, "Integer")) return KN_INT;
  if (na == 1 && sp_streq(m, "Float")) return KN_FLOAT;
  if (na == 1 && sp_streq(m, "String")) return KN_STR;
  if (na == 1 && sp_streq(m, "Array")) return KN_ARR;
  if (sp_streq(m, "Rational") || sp_streq(m, "Complex")) return KN_OTHER;
  if (na >= 1 && (sp_streq(m, "format") || sp_streq(m, "sprintf"))) return KN_STR;
  if (sp_streq(m, "puts") || sp_streq(m, "print")) return KN_NIL;
  if (sp_streq(m, "raise") || sp_streq(m, "fail")) return 0;
  if (sp_streq(m, "block_given?")) return KN_BOOL;
  return any;
}
/* the kinds of SEND value v, from its operands' kinds */
static Kd ir_send_kinds(Compiler *c, const NodeTable *nt, Ir *ir, const Kd *K, int v) {
  Kd any = { KN_ALL | KN_UNK, -2 }, r = { 0, -1 };
  IrVal *x = &ir->v[v];
  int node = x->node;
  const char *m = nt_kind(nt, node) == NK_CallNode ? nt_str(nt, node, "name") : nt_str(nt, node, "binary_operator");
  if (!m) return any;
  int fl = nt_kind(nt, node) == NK_CallNode ? x->evk : 0;
  int rv = x->op0 >= 0 ? ir->ops[x->op0] : -1;
  int na = x->nop - 1;
  Kd ak[40];
  for (int i = 0; i < na && i < 40; i++) {
    int o = ir->ops[x->op0 + 1 + i];
    if (o < 0) return any;
    ak[i] = K[o];
    if (!ak[i].k) return r;              /* an argument with no value yet */
  }
  if (rv < 0 && nt_kind(nt, node) != NK_CallNode) return any;
  if (rv < 0) {                            /* a call on self */
    if (fl & (SF_ODD | SF_BLOCK)) {
      if ((fl & SF_BLOCK) && (sp_streq(m, "lambda") || sp_streq(m, "proc")) &&
          !ir_names_has(g_ir_names.all, g_ir_names.nall, m)) { r.k = KN_OTHER; return r; }
      return any;
    }
    r.k = ir_self_rule(m, na);
    if (r.k & KN_UNK) r.cid = -2;
    return r;
  }
  if (fl & (SF_ODD | SF_BLOCK)) return any;
  if (ir_names_has(g_ir_names.core, g_ir_names.ncore, m)) return any;
  if (ir->v[rv].kind == IV_OTHER && ir->v[rv].node >= 0 && nt_kind(nt, ir->v[rv].node) == NK_ConstantReadNode)
    return ir_class_call_kinds(c, nt_str(nt, ir->v[rv].node, "name"), m);
  Kd rk = K[rv];
  if (!rk.k) return r;                     /* the receiver has no value yet */
  if (rk.k & KN_UNK) {
    /* A conversion gives one kind on every built-in receiver. When the
       program defines no method of that name, an unknown receiver gives it
       too. SP_IR_CONV_OFF=1 turns this rule off. */
    if (na == 0 && !(fl & SF_SAFE) && !getenv("SP_IR_CONV_OFF") && !ir_names_has(g_ir_names.all, g_ir_names.nall, m)) {
      if (sp_streq(m, "to_i")) { r.k = KN_INT; return r; }
      if (sp_streq(m, "to_f")) { r.k = KN_FLOAT; return r; }
      if (sp_streq(m, "to_sym")) { r.k = KN_SYM; return r; }
    }
    return any;
  }
  unsigned short rb = rk.k;
  if (fl & SF_SAFE) { r.k |= rb & KN_NIL; rb &= (unsigned short)~KN_NIL; }   /* `a&.m` on nil is nil */
  for (int b = 0; b < 12; b++) {
    if (!(rb & (1 << b))) continue;
    unsigned short k = ir_rule(1 << b, m, na, ak);
    if (k & KN_UNK) return any;
    r.k |= k;
  }
  return r;
}
/* the kinds of a literal */
static unsigned short ir_const_kinds(const NodeTable *nt, int node) {
  if (node < 0) return KN_NIL;
  switch (nt_kind(nt, node)) {
  case NK_IntegerNode: return nt_str(nt, node, "bigval") ? KN_BIG : KN_INT;
  case NK_FloatNode: return KN_FLOAT;
  case NK_StringNode: case NK_InterpolatedStringNode: case NK_XStringNode: case NK_InterpolatedXStringNode:
  case NK_SourceFileNode: return KN_STR;
  case NK_SymbolNode: case NK_InterpolatedSymbolNode: return KN_SYM;
  case NK_NilNode: return KN_NIL;
  case NK_TrueNode: return KN_TRUE;
  case NK_FalseNode: return KN_FALSE;
  case NK_ArrayNode: return KN_ARR;
  case NK_HashNode: return KN_HASH;
  case NK_SourceLineNode: return KN_INT;
  case NK_DefinedNode: return KN_STR | KN_NIL;
  default: return KN_OTHER;   /* a Range, a Regexp, a Rational, a lambda */
  }
}
/* The kinds of every value of an IR that has its expression values:
   from the literals, the rules and the parameters (any value) alone. */
static Kd *ir_own_kinds(CfgB *g, Ir *ir, const char *is_param) {
  int nv = ir->nv;
  const NodeTable *nt = g->nt;
  Kd *K = malloc(sizeof(Kd) * ((size_t)nv + 1));
  if (!K) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  Kd any = { KN_ALL | KN_UNK, -2 }, nil = { KN_NIL, -1 }, none = { 0, -1 };
  for (int x = 0; x < nv; x++) {
    IrVal *v = &ir->v[x];
    K[x] = none;
    if (v->kind == IV_ENTRY) K[x] = is_param[v->local] ? any : nil;
    else if (v->kind == IV_CONST) K[x].k = ir_const_kinds(nt, v->node);
    else if (v->kind == IV_RESCUED) { K[x].k = KN_OTHER | KN_OBJ; K[x].cid = -2; }
    else if (v->kind == IV_OTHER) {
      NodeKind nk = v->node >= 0 ? nt_kind(nt, v->node) : NK_NilNode;
      /* a jump gives no value where it is */
      if (!(nk == NK_ReturnNode || nk == NK_BreakNode || nk == NK_NextNode || nk == NK_RedoNode || nk == NK_RetryNode))
        K[x] = any;
    } else if (v->kind == IV_DEF && v->arg < 0) K[x] = any;
  }
  for (int changed = 1, round = 0; changed && round < 1000; round++) {
    changed = 0;
    for (int x = 0; x < nv; x++) {
      IrVal *v = &ir->v[x];
      Kd r = none;
      switch (v->kind) {
      case IV_DEF:
        if (v->arg < 0) continue;
        r = K[v->arg];
        break;
      case IV_PI:
        if (v->src < 0) continue;
        r = kd_refine(K[v->src], &g->pis[v->evk]);
        break;
      case IV_PHI:
        if (v->repl >= 0) continue;
        /* fallthrough */
      case IV_JOIN:
        for (int q = 0; q < v->nop; q++) { int o = ir->ops[v->op0 + q]; if (o >= 0) r = kd_meet(r, K[o]); }
        break;
      case IV_AND: case IV_OR: {
        int a = ir->ops[v->op0], b = ir->ops[v->op0 + 1];
        Kd ka = a >= 0 ? K[a] : any, kb = b >= 0 ? K[b] : any;
        /* `a && b`: a when it is nil or false, else b; `a || b` the other way */
        unsigned short keep = v->kind == IV_AND ? (KN_NIL | KN_FALSE) : (unsigned short)~(KN_NIL | KN_FALSE);
        Kd kl = ka; kl.k = (unsigned short)(ka.k & (keep | KN_UNK));
        if (!(kl.k & KN_OBJ)) kl.cid = -1;
        r = kl;
        if (ka.k & ~keep & ~KN_UNK) r = kd_meet(r, kb);
        else if (ka.k & KN_UNK) r = kd_meet(r, kb);
        break;
      }
      case IV_SEND:
        r = ir_send_kinds(g->c, nt, ir, K, x);
        break;
      default:
        continue;
      }
      if (r.k != K[x].k || r.cid != K[x].cid) { K[x] = r; changed = 1; }
    }
  }
  return K;
}

/* (analyze.c) the is_a? guard walk's class rule and its Array rule */
TyKind isa_narrow_type(Compiler *c, const char *cn, const Scope *s);
int isa_array_read_only(const char *nm);

static int st_ir_scopes, st_ir_reads, st_ir_narrowed, st_ir_pis;

/* the reads of scope s whose kinds prove a class the tree's guard walk
   would unbox to: the early pass */
static void ir_narrow_scope_early(Compiler *c, Scope *s, CfgB *g, Ir *ir, const int *par) {
  Kd *K = ir_solve_kinds(g, ir, 0);
  st_ir_pis += g->npis;
  for (int b = 0; b < g->nb; b++) {
    if (!ir->reached[b]) continue;
    CBlock *bl = &g->b[b];
    for (int e = 0; e < bl->nev; e++) {
      if (bl->ev_kind[e] != EV_READ) continue;
      int li = bl->ev_loc[e], r = bl->ev_node[e], x = bl->ev_val[e];
      if (li < 0 || x < 0 || r < 0 || r >= c->nt->count || g->captured[li]) continue;
      if (nt_kind(c->nt, r) != NK_LocalVariableReadNode) continue;
      st_ir_reads++;
      const char *cn = kd_class(c, K[x]);
      if (!cn) continue;
      LocalVar *lv = &s->locals[li];
      if (lv->type != TY_POLY) continue;
      TyKind t = isa_narrow_type(c, cn, s);
      if (t == TY_UNKNOWN) continue;
      int p = par ? par[r] : -1;
      if (t == TY_POLY_ARRAY) {
        /* a copy: only as the receiver of a method that reads it */
        if (p < 0 || nt_kind(c->nt, p) != NK_CallNode || nt_ref(c->nt, p, "receiver") != r ||
            !isa_array_read_only(nt_str(c->nt, p, "name"))) continue;
      } else if (p >= 0 && (nt_kind(c->nt, p) == NK_ArrayNode || nt_kind(c->nt, p) == NK_HashNode)) continue;
      if (cfg_statement_writes(c, par, r, lv->name)) continue;
      if (c->nilnarrow[r] != t) { comp_set_nilnarrow(c, r, t); st_ir_narrowed++; }
    }
  }
  ir_dump(c, s, g, ir, K, "early");
  free(K);
}

/* The early pass: in place of the is_a? guard walk, for every method and
   the top level. Answers the scopes it did (the walk does the rest), or
   NULL when it is off. */
char *ir_narrow_type_tests(Compiler *c) {
  if (ir_off()) return NULL;
  char *done = calloc((size_t)c->nscopes + 1, 1);
  if (!done) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  int *par = an_parent_map(c->nt);
  st_ir_scopes = st_ir_reads = st_ir_narrowed = st_ir_pis = 0;
  for (int si = 0; si < c->nscopes; si++) {
    Scope *s = &c->scopes[si];
    if (si != 0 && (s->def_node < 0 || nt_kind(c->nt, s->def_node) != NK_DefNode)) continue;
    if (s->body < 0 || s->cs_synth) continue;
    CfgB g; Ir ir;
    if (s->nlocals <= 0) { done[si] = 1; continue; }
    if (!ir_scope_build(c, s, &g, &ir, 0)) { ir_scope_free(&g, &ir); continue; }
    ir_narrow_scope_early(c, s, &g, &ir, par);
    ir_scope_free(&g, &ir);
    done[si] = 1;
    st_ir_scopes++;
  }
  free(par);
  if (ir_log()) fprintf(stderr, "spinel-ir: early: scopes=%d pis=%d reads=%d narrowed=%d\n",
                        st_ir_scopes, st_ir_pis, st_ir_reads, st_ir_narrowed);
  return done;
}

/* The late pass: a read of a poly local whose every write that reaches it
   stored one plain Integer or Float (or nil, where a test took the nil
   out) unboxes to it. */
int cfg_narrow_reads(Compiler *c) {
  if (cfg_disabled()) return 0;
  if (ir_off()) return cfg_narrow_reads_classic(c);
  st_narrowed = 0;
  int *par = an_parent_map(c->nt);
  for (int si = 0; si < c->nscopes; si++) {
    Scope *s = &c->scopes[si];
    if (si != 0 && (s->def_node < 0 || nt_kind(c->nt, s->def_node) != NK_DefNode)) continue;
    CfgB g; Ir ir;
    if (!ir_scope_build(c, s, &g, &ir, 1)) { ir_scope_free(&g, &ir); continue; }
    Kd *K = ir_solve_kinds(&g, &ir, 1);
    for (int b = 0; b < g.nb; b++) {
      if (!ir.reached[b]) continue;
      CBlock *bl = &g.b[b];
      for (int e = 0; e < bl->nev; e++) {
        if (bl->ev_kind[e] != EV_READ) continue;
        int li = bl->ev_loc[e], r = bl->ev_node[e], x = bl->ev_val[e];
        if (li < 0 || x < 0 || r < 0 || r >= c->nt->count || g.captured[li]) continue;
        if (nt_kind(c->nt, r) != NK_LocalVariableReadNode) continue;
        TyKind t = K[x].k == KN_INT ? TY_INT : K[x].k == KN_FLOAT ? TY_FLOAT : TY_UNKNOWN;
        if (t == TY_UNKNOWN) continue;
        LocalVar *lv = &s->locals[li];
        if (lv->type != TY_POLY || lv->is_cell || lv->proc_rebinds) continue;
        if (c->nilnarrow[r] != TY_UNKNOWN) continue;
        if (direct_literal_elem(c, par, r)) continue;
        /* the prelude's methods convert their arguments through #to_int and
           friends at sites the reachability pass finds by their boxed reads:
           their reads stay boxed */
        if (nt_int(c->nt, r, "node_bi", 0) != 0) continue;
        if (cfg_statement_writes(c, par, r, lv->name)) continue;
        comp_set_nilnarrow(c, r, t);
        st_narrowed++;
      }
    }
    ir_dump(c, s, &g, &ir, K, "late");
    free(K);
    ir_scope_free(&g, &ir);
  }
  free(par);
  if (cfg_log() || ir_log()) fprintf(stderr, "spinel-ir: late: narrowed=%d\n", st_narrowed);
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
  int *resc, nresc, cresc;   /* the targets rescue arms bind (`rescue => e`) */
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
static int ssa_blocks_off(void) {
  static int v = -1;
  if (v < 0) v = getenv("SP_NO_SSA_BLOCKS") != NULL || ir_off();
  return v;
}
static void ssa_collect_in(Compiler *c, Scope *s, int n, SsaNames *t, int in_closure, int opaque, int depth, int blk_on);
static void ssa_collect(Compiler *c, Scope *s, int n, SsaNames *t, int in_closure, int opaque, int depth) {
  ssa_collect_in(c, s, n, t, in_closure, opaque, depth, 0);
}
/* blk_on: n is a block the graph runs inside its call. Its own locals
   (the parameters, and a local first written in it) keep one slot. */
static void ssa_collect_in(Compiler *c, Scope *s, int n, SsaNames *t, int in_closure, int opaque, int depth, int blk_on) {
  if (n < 0) return;
  if (depth > 2000) { t->method_bad = 1; return; }
  const NodeTable *nt = c->nt;
  NodeKind k = nt_kind(nt, n);
  if (k == NK_DefNode || k == NK_ClassNode || k == NK_ModuleNode || k == NK_SingletonClassNode) return;
  if ((k == NK_BlockNode && !blk_on) || k == NK_LambdaNode) in_closure = 1;
  if (k == NK_BlockNode && blk_on) {
    const char *ls = nt_str(nt, n, "locals");
    for (const char *p = ls; p && *p; ) {
      const char *e = strchr(p, ',');
      size_t l = e ? (size_t)(e - p) : strlen(p);
      char nm[256];
      if (l > 0 && l < sizeof nm) { memcpy(nm, p, l); nm[l] = 0; int ix = ssa_name_add(t, nm); t->bad[ix] = 1; }
      if (!e) break;
      p = e + 1;
    }
  }
  int gblk = -1;
  if (!in_closure && !ssa_blocks_off() && (k == NK_CallNode || k == NK_SuperNode || k == NK_ForwardingSuperNode) &&
      cfg_block_on_graph(c, n, 0))
    gblk = nt_ref(nt, n, "block");
  if (k == NK_DefinedNode || k == NK_MatchWriteNode) opaque = 1;
  if (k == NK_RescueNode && nt_ref(nt, n, "reference") >= 0) {
    if (t->nresc >= t->cresc) {
      t->cresc = t->cresc ? t->cresc * 2 : 8;
      t->resc = realloc(t->resc, sizeof(int) * (size_t)t->cresc);
      if (!t->resc) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
    }
    t->resc[t->nresc++] = nt_ref(nt, n, "reference");
  }
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
    ssa_collect_in(c, s, ch, t, in_closure, opaque || (ch >= 0 && ch == pat), depth + 1, ch >= 0 && ch == gblk);
  }
  for (int i = 0; i < nt_num_arrs(nt, n); i++) {
    int an = 0; const int *av = nt_arr_at(nt, n, i, &an);
    for (int q = 0; q < an; q++) ssa_collect_in(c, s, av[q], t, in_closure, opaque, depth + 1, 0);
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
static int st_ssa_methods, st_ssa_split, st_ssa_renamed, st_ssa_phis, st_ssa_ir_whole;
/* SP_IR_TYPES_OFF: the SSA rename decides without the IR's own types */
static int ir_types_off(void) {
  static int v = -1;
  if (v < 0) v = getenv("SP_IR_TYPES_OFF") != NULL;
  return v;
}

/* Each split made, so ssa_check_splits can read the types its slots got.
   A split is kept only when every slot of it is a plain value (Integer,
   Float, String, Symbol, true/false, nil). An Array or Hash slot is where
   Spinel's typed containers can refuse an element a poly slot took, so a
   split that types one is undone: the compile runs again with that local
   left whole (SP_SSA_VETO). */
typedef struct { int scope; char *key; char **names; int nnames; int entry_nil; } SsaSplit;
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
/* marks the names of t that a def's parameters bind */
static void ssa_param_names(const NodeTable *nt, int n, SsaNames *t, char *is_param, int depth) {
  if (n < 0 || depth > 64) return;
  const char *ty = nt_type(nt, n);
  if (ty && strstr(ty, "ParameterNode")) {
    const char *nm = nt_str(nt, n, "name");
    if (nm) for (int i = 0; i < t->n; i++) if (sp_streq(t->nm[i], nm)) is_param[i] = 1;
  }
  for (int i = 0; i < nt_num_refs(nt, n); i++) ssa_param_names(nt, nt_ref_at(nt, n, i), t, is_param, depth + 1);
  for (int i = 0; i < nt_num_arrs(nt, n); i++) {
    int an = 0; const int *av = nt_arr_at(nt, n, i, &an);
    for (int q = 0; q < an; q++) ssa_param_names(nt, av[q], t, is_param, depth + 1);
  }
}

/* The kinds of value each slot holds by the IR's own types, kept until
   the types are known (ir_types_report compares them, SP_IR_TYPES) */
typedef struct { int scope; char *name; Kd k; int unk, ndef, param; } IrSlot;
static IrSlot *g_ir_slots;
static int g_nir_slots, g_cir_slots;
static void ir_slot_note(int scope, const char *name, Kd k, int known, int param) {
  static int on = -1;
  if (on < 0) on = getenv("SP_IR_TYPES") != NULL;
  if (!on) return;
  IrSlot *sl = NULL;
  for (int i = g_nir_slots - 1; i >= 0 && g_ir_slots[i].scope == scope; i--)
    if (sp_streq(g_ir_slots[i].name, name)) { sl = &g_ir_slots[i]; break; }
  if (!sl) {
    if (g_nir_slots >= g_cir_slots) {
      g_cir_slots = g_cir_slots ? g_cir_slots * 2 : 64;
      g_ir_slots = realloc(g_ir_slots, sizeof(IrSlot) * (size_t)g_cir_slots);
      if (!g_ir_slots) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
    }
    sl = &g_ir_slots[g_nir_slots++];
    sl->scope = scope; sl->name = strdup(name); sl->k.k = 0; sl->k.cid = -1; sl->unk = 0; sl->ndef = 0; sl->param = 0;
    if (!sl->name) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  }
  sl->ndef++;
  if (param) sl->param = 1;
  if (!known) { sl->unk = 1; return; }
  sl->k = kd_meet(sl->k, k);
}

/* The plain kinds of value one slot of Spinel's can hold. Two of them in
   one slot make it poly (an Integer and a String); nil joins an Integer, a
   Float or a String in their slots, not true/false or a Symbol. */
enum { CAT_BOOL = 1, CAT_INT = 2, CAT_FLOAT = 4, CAT_STR = 8, CAT_SYM = 16 };
static int kd_cats(unsigned short k) {
  return (k & KN_BOOL ? CAT_BOOL : 0) | (k & KN_INTS ? CAT_INT : 0) | (k & KN_FLOAT ? CAT_FLOAT : 0) |
         (k & KN_STR ? CAT_STR : 0) | (k & KN_SYM ? CAT_SYM : 0);
}
static int cat_count(int c) { int n = 0; for (; c; c &= c - 1) n++; return n; }
static const char *cat_names(int c, char *buf, size_t cap) {
  static const char *const nm[] = { "true/false", "Integer", "Float", "String", "Symbol" };
  buf[0] = 0;
  for (int b = 0; b < 5; b++) if (c & (1 << b)) {
    size_t l = strlen(buf);
    snprintf(buf + l, cap - l, "%s%s", l ? " and " : "", nm[b]);
  }
  return buf;
}
/* Why the IR's own types show that splitting local li gives a slot that
   is not a plain value (ssa_check_splits would undo the split after
   typing), or NULL: a web where a write stores an Array, a Hash, an
   object or another boxed value, or where two plain kinds meet that no
   one slot holds. Writes the IR has no kind for prove nothing. */
static const char *ssa_ir_boxed(Ir *ir, const Kd *K, int *uf, const int *dl, const int *dl_off, int li,
                                int param, char *why, size_t cap) {
  int entry_web = ssa_find(uf, li);
  for (int q = dl_off[li]; q < dl_off[li + 1]; q++) {
    int r = ssa_find(uf, dl[q]), seen = 0;
    for (int q2 = dl_off[li]; q2 < q; q2++) if (ssa_find(uf, dl[q2]) == r) { seen = 1; break; }
    if (seen) continue;
    int cats = 0, nil = 0;
    for (int q2 = q; q2 < dl_off[li + 1]; q2++) {
      int d = dl[q2];
      if (ssa_find(uf, d) != r || ir->v[d].kind != IV_DEF || !ir->reached[ir->v[d].block]) continue;
      unsigned short k = K[d].k;
      if (!k || (k & KN_UNK)) continue;
      if (k & KN_BOXY) {
        snprintf(why, cap, "a write stores %s", k & KN_ARR ? "an Array" : k & KN_HASH ? "a Hash" :
                 k & KN_OBJ ? "an object" : "a boxed value (a Range, a Rational, an exception)");
        return why;
      }
      cats |= kd_cats(k);
      if (k & KN_NIL) nil = 1;
    }
    char b[96];
    if (cat_count(cats) >= 2) { snprintf(why, cap, "one slot gets %s", cat_names(cats, b, sizeof b)); return why; }
    if (nil && (cats == CAT_BOOL || cats == CAT_SYM)) { snprintf(why, cap, "one slot gets nil and %s", cat_names(cats, b, sizeof b)); return why; }
    /* a true/false slot has no nil: a read before the first write (`x =
       true if c; p x`) would see false */
    if (r == entry_web && !param && ir->v[li].live && cats == CAT_BOOL) {
      snprintf(why, cap, "a true/false slot that a read can see before its first write");
      return why;
    }
  }
  return NULL;
}

static void ssa_method(Compiler *c, Scope *s) {
  const NodeTable *nt = c->nt;
  int body = s->body >= 0 ? s->body : nt_ref(nt, s->def_node, "body");
  if (body < 0) return;
  SsaNames t; memset(&t, 0, sizeof t);
  ssa_collect(c, s, body, &t, 0, 0, 0);
  if (t.method_bad || t.n == 0) {
    for (int i = 0; i < t.n; i++) free((char *)t.nm[i]);
    free(t.nm); free(t.bad); free(t.nodes); free(t.node_name); free(t.pw); free(t.pr); free(t.resc);
    return;
  }
  /* a parameter's entry definition is the argument, any other's is nil:
     either way the web it is in keeps the name */
  CfgB g; memset(&g, 0, sizeof g);
  g.c = c; g.nt = nt; g.s = s; g.si = s->idx; g.ssa = 1;
  g.blocks = !ssa_blocks_off();
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
  /* the IR's values: one entry value per name (the argument, or nil
     before any write), one per write, and the phis the reads need */
  Ir ir; memset(&ir, 0, sizeof ir);
  int *uf = NULL, *dl_off = NULL, *dl = NULL;
  char *has_read = NULL;
  Kd *K = NULL;
  char *is_param = ir_xalloc((size_t)nn + 1);
  ssa_param_names(nt, nt_ref(nt, s->def_node, "parameters"), &t, is_param, 0);
  int nvals = 0;
  { /* a huge method is left as it is (the bound the reaching definitions
       had: a bit per write, per block) */
    size_t ndef = (size_t)nn;
    for (int b = 0; b < g.nb; b++)
      for (int e = 0; e < g.b[b].nev; e++) if (g.b[b].ev_kind[e] != EV_READ) ndef++;
    if ((size_t)g.nb * ((ndef + 63) / 64) > ((size_t)1 << 22)) goto done; }
  if (!ir_build(&g, nn, &ir)) goto done;
  nvals = ir.nv;
  uf = malloc(sizeof(int) * ((size_t)nvals + 1));
  has_read = ir_xalloc((size_t)nvals + 1);
  if (!uf) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  for (int i = 0; i < nvals; i++) uf[i] = i;
  /* each name's entry value and writes, in the order the values were made */
  dl_off = ir_xalloc(sizeof(int) * ((size_t)nn + 2));
  dl = malloc(sizeof(int) * ((size_t)nvals + 1));
  if (!dl) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  for (int i = 0; i < nvals; i++)
    if (ir.v[i].kind == IV_ENTRY || ir.v[i].kind == IV_DEF) dl_off[ir.v[i].local + 1]++;
  for (int i = 0; i < nn; i++) dl_off[i + 1] += dl_off[i];
  { int *fill = ir_xalloc(sizeof(int) * ((size_t)nn + 1));
    for (int i = 0; i < nvals; i++)
      if (ir.v[i].kind == IV_ENTRY || ir.v[i].kind == IV_DEF) dl[dl_off[ir.v[i].local] + fill[ir.v[i].local]++] = i;
    free(fill); }
  /* out of SSA, a phi and its operands share one slot, and so do a pi and
     the value it narrows: a web is one class of these */
  for (int i = 0; i < nvals; i++) {
    IrVal *v = &ir.v[i];
    if (!v->live) continue;
    if (v->kind == IV_PHI && v->repl < 0) {
      st_ssa_phis++;
      if (ssa_log()) fprintf(stderr, "spinel-ssa:   `%s` meets at a phi of %d values\n", t.nm[v->local], v->nop);
      for (int k = 0; k < v->nop; k++) if (ir.ops[v->op0 + k] >= 0) ssa_union(uf, i, ir.ops[v->op0 + k]);
    }
    if (v->kind == IV_PI && v->src >= 0) ssa_union(uf, i, v->src);
  }
  /* the value each read sees and each write makes, by node */
  for (int b = 0; b < g.nb; b++) {
    CBlock *bl = &g.b[b];
    for (int e = 0; e < bl->nev; e++) {
      int node = bl->ev_node[e];
      if (node >= g_ssa_rd_cap || bl->ev_loc[e] < 0) continue;
      if (bl->ev_kind[e] == EV_READ) g_ssa_rd[node] = bl->ev_val[e];
      else if (bl->ev_kind[e] != EV_PI) g_ssa_wd[node] = bl->ev_val[e];
    }
  }
  /* the IR's own types: a value for each step of what each write stores,
     and the kinds of every value from those alone */
  if (!ir_types_off()) {
    ir_add_exprs(&ir, nt, g_ssa_rd, g_ssa_wd, g_ssa_rd_cap, t.resc, t.nresc);
    K = ir_own_kinds(&g, &ir, is_param);
    ir_dump(c, s, &g, &ir, K, "SSA rename, before typing");
  }
  /* `x += 1`, `x ||= v`: one node reads and writes, so one name */
  for (int b = 0; b < g.nb; b++) {
    CBlock *bl = &g.b[b];
    for (int e = 0; e < bl->nev; e++) {
      int dd = bl->ev_kind[e] != EV_READ && bl->ev_kind[e] != EV_PI ? bl->ev_val[e] : -1, node = bl->ev_node[e];
      if (dd < 0 || node >= g_ssa_rd_cap) continue;
      NodeKind nk = nt_kind(nt, node);
      if (nk != NK_LocalVariableOperatorWriteNode && nk != NK_LocalVariableOrWriteNode &&
          nk != NK_LocalVariableAndWriteNode) continue;
      if (g_ssa_rd[node] >= 0) ssa_union(uf, dd, g_ssa_rd[node]);
    }
  }
  /* a write whose value reads the local: the write joins the web of the
     values that read sees (ssa_value_reads) */
  int selfread_off = getenv("SP_SSA_SELFREAD_OFF") != NULL;
  for (int q = 0; q < t.np; q++) {
    int w = t.pw[q] < g_ssa_rd_cap ? g_ssa_wd[t.pw[q]] : -1;
    int r = t.pr[q] < g_ssa_rd_cap ? g_ssa_rd[t.pr[q]] : -1;
    if (w < 0 || r < 0) continue;
    /* The value the write stores is never the object the read saw when the
       IR's own types say it is a number, a Symbol, true, false or nil
       (`n = n.to_i`): nothing can be stored through it, so the write may
       start a web of its own. SP_SSA_SELFREAD_OFF=1 keeps the old rule. */
    if (K && !selfread_off && K[w].k && !(K[w].k & ~(KN_NIL | KN_BOOL | KN_INT | KN_BIG | KN_FLOAT | KN_SYM))) {
      if (getenv("SP_SSA_LOG")) fprintf(stderr, "spinel-ssa: a write that reads its own local stores a plain value: not joined\n");
      continue;
    }
    ssa_union(uf, w, r);
  }
  for (int b = 0; b < g.nb; b++)
    for (int e = 0; e < g.b[b].nev; e++)
      if (g.b[b].ev_kind[e] == EV_READ && g.b[b].ev_val[e] >= 0)
        has_read[ssa_find(uf, g.b[b].ev_val[e])] = 1;

  /* the webs of each name, and their names */
  {
    char **web_name = calloc((size_t)nvals + 1, sizeof(char *));
    char *keeps = calloc((size_t)nvals + 1, 1);
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
      if (roots < 2) {
        if (K) for (int q = dl_off[li]; q < dl_off[li + 1]; q++)
          if (ir.v[dl[q]].kind == IV_DEF || is_param[li]) ir_slot_note(s->idx, t.nm[li], K[dl[q]], !(K[dl[q]].k & KN_UNK), ir.v[dl[q]].kind == IV_ENTRY);
        continue;
      }
      /* the entry definition's web keeps the name (a parameter is that
         name); when no read sees it, the first write's web joins it */
      int r0 = ssa_find(uf, li);
      keeps[r0] = 1;
      if (!has_read[r0]) {
        int best = -1;
        for (int q = dl_off[li]; q < dl_off[li + 1]; q++)
          if (ir.v[dl[q]].node >= 0 && (best < 0 || ir.v[dl[q]].node < ir.v[best].node)) best = dl[q];
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
      /* The IR's own types, before typing: a split that gives a slot no
         plain value would be undone after typing (ssa_check_splits), and
         the compile would run again. The local stays whole now. */
      { char why[128];
        if (k && K && ssa_ir_boxed(&ir, K, uf, dl, dl_off, li, is_param[li], why, sizeof why)) {
          for (int q = dl_off[li]; q < dl_off[li + 1]; q++) {
            int r = ssa_find(uf, dl[q]);
            free(web_name[r]); web_name[r] = NULL;
          }
          t.bad[li] = 1;
          st_ssa_ir_whole++;
          if (ssa_log()) fprintf(stderr, "spinel-ssa: %s: the IR's own types: %s: one slot, before typing\n", key, why);
          for (int q = dl_off[li]; q < dl_off[li + 1]; q++)
            if (ir.v[dl[q]].kind == IV_DEF || is_param[li]) ir_slot_note(s->idx, t.nm[li], K[dl[q]], !(K[dl[q]].k & KN_UNK), ir.v[dl[q]].kind == IV_ENTRY);
          continue;
        } }
      if (K) for (int q = dl_off[li]; q < dl_off[li + 1]; q++) {
        int d = dl[q], r = ssa_find(uf, d);
        if (ir.v[d].kind != IV_DEF && !is_param[li]) continue;   /* (nil before the first write: Spinel's types leave it out) */
        ir_slot_note(s->idx, keeps[r] || !web_name[r] ? t.nm[li] : web_name[r], K[d], !(K[d].k & KN_UNK), ir.v[d].kind == IV_ENTRY);
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
        /* the kept name's slot is nil before the first write when a read
           can see the entry */
        sp->entry_nil = !is_param[li] && ir.v[li].live;
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
                fprintf(stderr, "%s%s", any ? "," : "", ir.v[dl[q2]].node < 0 ? "entry" : "");
                if (ir.v[dl[q2]].node >= 0) fprintf(stderr, "line %d", ssa_line(c, ir.v[dl[q2]].node));
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
          int li = bl->ev_loc[e];
          if (li < 0 || t.bad[li] || bl->ev_kind[e] == EV_PI || !ssa_local_node(nt_kind(nt, node))) continue;
          int dd = bl->ev_val[e];
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
      for (int i = 0; i < nvals; i++) if (web_name[i]) tot += strlen(web_name[i]) + 1;
      char *nl = malloc(tot + 1);
      if (!nl) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
      snprintf(nl, tot + 1, "%s", ol ? ol : "");
      for (int i = 0; i < nvals; i++)
        if (web_name[i]) { if (nl[0]) strcat(nl, ","); strcat(nl, web_name[i]); }
      if (ol) nt_set_str((NodeTable *)nt, s->def_node, "locals", nl);
      free(nl);
    }
    for (int i = 0; i < nvals; i++) free(web_name[i]);
    free(web_name); free(keeps);
  }
done:
  for (int b = 0; b < g.nb; b++) {
    CBlock *bl = &g.b[b];
    for (int e = 0; e < bl->nev; e++) if (bl->ev_node[e] < g_ssa_rd_cap) g_ssa_rd[bl->ev_node[e]] = g_ssa_wd[bl->ev_node[e]] = -1;
    free(bl->ev_kind); free(bl->ev_node); free(bl->succ); free(bl->ev_val); free(bl->ev_loc); free(bl->ev_aux);
  }
  free(g.b);
  ir_free(&ir);
  free(uf); free(has_read); free(dl_off); free(dl);
  for (int i = 0; i < t.n; i++) free((char *)t.nm[i]);
  free(t.nm); free(t.bad); free(t.nodes); free(t.node_name); free(t.pw); free(t.pr); free(t.resc);
  free(K); free(is_param);
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
  st_ssa_methods = st_ssa_split = st_ssa_renamed = st_ssa_phis = st_ssa_ir_whole = 0;
  ir_names_build(c);
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
  ir_names_free();
  if (ssa_log()) fprintf(stderr, "spinel-ssa: methods=%d phis=%d locals-split=%d nodes-renamed=%d ir-whole=%d\n",
                         st_ssa_methods, st_ssa_phis, st_ssa_split, st_ssa_renamed, st_ssa_ir_whole);
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
/* the nodes inside a block that can run after its call returns */
static void ssa_mark_tree(const NodeTable *nt, int n, char *m, int depth) {
  if (n < 0 || depth > 2000 || m[n]) return;
  m[n] = 1;
  for (int i = 0; i < nt_num_refs(nt, n); i++) ssa_mark_tree(nt, nt_ref_at(nt, n, i), m, depth + 1);
  for (int i = 0; i < nt_num_arrs(nt, n); i++) {
    int an = 0; const int *av = nt_arr_at(nt, n, i, &an);
    for (int q = 0; q < an; q++) ssa_mark_tree(nt, av[q], m, depth + 1);
  }
}
static char *ssa_later_nodes(Compiler *c) {
  const NodeTable *nt = c->nt;
  char *m = calloc((size_t)nt->count + 1, 1);
  if (!m) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  static const NodeKind kinds[3] = { NK_CallNode, NK_SuperNode, NK_ForwardingSuperNode };
  for (int q = 0; q < 3; q++)
    NT_FOREACH_KIND(nt, kinds[q], id) {
      int blk = nt_ref(nt, id, "block");
      if (blk < 0 || nt_kind(nt, blk) != NK_BlockNode || m[blk]) continue;
      if (an_block_outlives_call(c, id)) ssa_mark_tree(nt, blk, m, 0);
    }
  return m;
}
/* SP_IR_TYPES=1 prints, once the types are known, how many slots the
   IR's own types give a kind and how many of those match the type
   Spinel's inference gave the slot (=2: each slot that does not match) */
static const char *ty_cat(TyKind t) {
  if (t == TY_NIL) return "nil";
  if (t == TY_INT || t == TY_BIGINT) return "Integer";
  if (t == TY_FLOAT) return "Float";
  if (t == TY_STRING || t == TY_STRBUF) return "String";
  if (t == TY_SYMBOL) return "Symbol";
  if (t == TY_BOOL) return "true/false";
  if (t == TY_POLY) return "poly";
  if (t == TY_UNKNOWN) return "unknown";
  if (ty_is_array(t) || ty_is_obj_array(t) || t == TY_POLY_ARRAY) return "Array";
  if (ty_is_hash(t)) return "Hash";
  if (ty_is_object(t)) return "Object";
  return "other";
}
static const char *kd_cat(Kd k) {
  unsigned short b = k.k & (unsigned short)~KN_NIL;
  if (k.k & KN_BOXY) {
    if (b == KN_ARR) return "Array";
    if (b == KN_HASH) return "Hash";
    if (b == KN_OBJ) return "Object";
    if (b && !(b & ~(KN_OTHER | KN_OBJ))) return "other";
    return "poly";
  }
  int cats = kd_cats(k.k);
  if (cat_count(cats) >= 2) return "poly";
  if ((k.k & KN_NIL) && (cats == CAT_BOOL || cats == CAT_SYM)) return "poly";
  if (!cats) return "nil";
  if (cats == CAT_BOOL) return "true/false";
  if (cats == CAT_INT) return "Integer";
  if (cats == CAT_FLOAT) return "Float";
  if (cats == CAT_STR) return "String";
  return "Symbol";
}
static void ir_types_report(Compiler *c) {
  const char *on = getenv("SP_IR_TYPES");
  if (!on) return;
  int slots = 0, typed = 0, agree = 0, unknown = 0, novalue = 0, boxed = 0, finer = 0, params = 0;
  for (int i = 0; i < g_nir_slots; i++) {
    IrSlot *sl = &g_ir_slots[i];
    Scope *sc = sl->scope >= 0 && sl->scope < c->nscopes ? &c->scopes[sl->scope] : NULL;
    if (!sc || !sc->reachable) continue;
    LocalVar *lv = scope_local(sc, sl->name);
    if (!lv) continue;
    if (sl->param) { params++; continue; }
    slots++;
    if (sl->unk) { unknown++; continue; }
    if (!sl->k.k) { novalue++; continue; }
    typed++;
    const char *a = kd_cat(sl->k), *b = ty_cat(lv->type);
    if (sp_streq(a, b) || (sp_streq(a, "other") && sp_streq(b, "Object"))) { agree++; continue; }
    /* both say the slot is no plain value (an Array, poly): the same split */
    int pa = !sp_streq(a, "poly") && !sp_streq(a, "Array") && !sp_streq(a, "Hash") && !sp_streq(a, "Object") && !sp_streq(a, "other");
    int pb = ssa_plain(lv->type);
    if (!pa && !pb) { boxed++; continue; }
    /* the IR has one plain kind where Spinel's types have poly */
    if (pa && (lv->type == TY_POLY || (pb && sp_streq(a, "nil")))) finer++;
    if (atoi(on) >= 2)
      fprintf(stderr, "spinel-ir-types: %s:%s: the IR's own types say %s, Spinel's types say %s (%s)\n",
              sc->name ? sc->name : "?", sl->name, a, b, ty_name(lv->type));
  }
  fprintf(stderr, "spinel-ir-types: slots=%d typed=%d agree=%d both-boxed=%d ir-finer=%d conflict=%d unknown=%d no-value=%d params=%d\n",
          slots, typed, agree, boxed, finer, typed - agree - boxed - finer, unknown, novalue, params);
}

void ssa_check_splits(Compiler *c) {
  ir_types_report(c);
  if (g_nssa_splits == 0) return;
  /* a split read or written inside a block that can run after its call
     (the graph ran it inside the call), or of a slot a proc shares */
  char *later_bad = calloc((size_t)g_nssa_splits + 1, 1);
  if (!later_bad) { fprintf(stderr, "spinel: out of memory\n"); exit(1); }
  if (!ssa_blocks_off()) {
    char *later = ssa_later_nodes(c);
    static const NodeKind lk[6] = { NK_LocalVariableReadNode, NK_LocalVariableWriteNode, NK_LocalVariableTargetNode,
      NK_LocalVariableOperatorWriteNode, NK_LocalVariableOrWriteNode, NK_LocalVariableAndWriteNode };
    for (int q = 0; q < 6; q++)
      NT_FOREACH_KIND(c->nt, lk[q], id) {
        if (!later[id]) continue;
        const char *nm = nt_str(c->nt, id, "name");
        if (!nm) continue;
        for (int i = 0; i < g_nssa_splits; i++) {
          SsaSplit *sp = &g_ssa_splits[i];
          if (later_bad[i] || c->nscope[id] != sp->scope) continue;
          for (int z = 0; z < sp->nnames; z++) if (sp_streq(sp->names[z], nm)) { later_bad[i] = 1; break; }
        }
      }
    free(later);
  }
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
    const char *why = NULL, *later = NULL;
    for (int q = 0; q < sp->nnames && !why; q++) {
      LocalVar *lv = scope_local(sc, sp->names[q]);
      if (!lv || !ssa_plain(lv->type)) why = sp->names[q];
      else if (lv->is_cell || lv->proc_rebinds || lv->cell_outlives) { why = sp->names[q]; later = "a proc shares it"; }
    }
    if (!why && later_bad[i]) { why = sp->names[0]; later = "a block that can run after its call uses it"; }
    /* a true/false slot has no nil: a read the entry reaches would see false */
    if (!why && sp->entry_nil) {
      LocalVar *lv = scope_local(sc, sp->names[0]);
      if (lv && lv->type == TY_BOOL) { why = sp->names[0]; later = "a true/false slot that a read can see before its first write"; }
    }
    if (!why) continue;
    if (ssa_log()) {
      LocalVar *lv = scope_local(sc, why);
      if (later) fprintf(stderr, "spinel-ssa: %s: %s: one slot again\n", sp->key, later);
      else fprintf(stderr, "spinel-ssa: %s: slot `%s` is %s, not a plain value: one slot again\n",
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
  free(later_bad);
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
