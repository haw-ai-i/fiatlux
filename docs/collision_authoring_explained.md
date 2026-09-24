1# How the Ladder Collision Was Authored

The Omniverse ladder packs ship as **rendering geometry only — zero PhysX collision**, so a
robot would clip straight through them. This doc explains in detail how we added collision,
done by `scripts/omniverse/omniverse_ladder_collision.py` — purely
with USD metadata (no GPU, no geometry baking).

## Core idea: an overlay layer, not editing the original

Instead of modifying the pack's `<name>.usd`, the script writes a separate
**`<name>_collision.usd`** that **sublayers** the original and adds collision on top via `over`
(override) prims. The original is never touched; the overlay just *annotates* its meshes with
physics. The result looks like:

```usda
#usda 1.0
(
    subLayers = [ @./tiltandrollladder_a01.usd@ ]   # pulls in the original (untouched)
    defaultPrim = "RootNode"
    upAxis = "Z"
    metersPerUnit = 1
)

over "RootNode"
{
    def Scope "PhysicsMaterials"
    {
        def Material "HighFriction" ( prepend apiSchemas = ["PhysicsMaterialAPI"] )
        {
            float physics:staticFriction = 1.2
            float physics:dynamicFriction = 1
            float physics:restitution = 0
        }
    }
    over "..._inst" ( instanceable = false )
    {
        over "SM_..._01"
        {
            over "M_..._Paint" ( prepend apiSchemas =
                ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "MaterialBindingAPI"] )
            {
                uniform token physics:approximation = "convexDecomposition"
                rel material:binding:physics = </RootNode/PhysicsMaterials/HighFriction>
            }
        }
    }
}
```

## Step by step, per ladder

**1. Open the original, read its stage info.** Default prim (`/World` or `/RootNode`),
`upAxis`, and `metersPerUnit` (cm vs m). The overlay copies these so it composes correctly.

**2. Create the overlay stage and sublayer the original.**
```python
stage = Usd.Stage.CreateNew("<name>_collision.usd")
stage.GetRootLayer().subLayerPaths.append("./<name>.usd")
```

**3. De-instance (SimReady only).** SimReady assets mark sub-trees `instanceable=true`, which
turns their meshes into read-only *instance proxies* you can't author on. The script walks the
tree and overrides them to expose the real meshes:
```python
for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
    if p.IsInstance() or p.IsInstanceable():
        stage.OverridePrim(p.GetPath()).SetInstanceable(False)
```

**4. Create a high-friction physics material**, inside the default-prim scope so the binding
survives if the asset is later referenced elsewhere:
```python
mat = UsdShade.Material.Define(stage, "<defaultPrim>/PhysicsMaterials/HighFriction")
pm  = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
pm.CreateStaticFrictionAttr(1.2); pm.CreateDynamicFrictionAttr(1.0); pm.CreateRestitutionAttr(0.0)
```

**5. For every Mesh, attach collision** via an `over` prim:
```python
over = stage.OverridePrim(mesh.GetPath())
UsdPhysics.CollisionAPI.Apply(over)                       # "this mesh is a collider"
UsdPhysics.MeshCollisionAPI.Apply(over).CreateApproximationAttr("convexDecomposition")
UsdShade.MaterialBindingAPI.Apply(over).Bind(mat, ..., "physics")   # grip
```

**6. Keep it static.** No `RigidBodyAPI` is applied → a fixed collider, so the ladder
(including the wheeled tilt-and-roll ones) won't fall or roll under load.

**7. Set default prim + save.**

## The key choice: `convexDecomposition`

`physics:approximation = "convexDecomposition"` is the most important part — it tells PhysX
*how* to turn each mesh into a collision shape:

- **convexHull** (naive) shrink-wraps the whole ladder into one solid convex blob — **filling
  the gaps between the steps**, so there's nothing to grip or stand between. Bad for climbing.
- **convexDecomposition** breaks the mesh into **many convex pieces** that follow the real
  shape — **keeping the gaps between rungs/steps open**. This is what makes it climbable.

> **This doc describes the original convexDecomposition authoring. The script now defaults to `SDF`**
> (signed distance field) — the exact-surface collider that also keeps *concave* features (the
> C-channel rail groove) open, which convexDecomposition cannot. convexDecomposition (tuned:
> `maxConvexHulls`, `hullVertexLimit`, `voxelResolution`, `shrinkWrap`) is kept as the fallback for
> designs whose mesh is wound inside-out (SDF would be inside-out there). It also authors a ~6 mm
> contact offset.

## When the actual collision geometry is created

The USD only stores the *directive* (`apiSchemas` + the `convexDecomposition` token). The real
collision shapes are **cooked by PhysX at load time** in Isaac Sim. That's why authoring needs
no GPU, and why the verify step (loading into PhysX) is what actually proves it works.

## Making it movable: the `_collision_rigid.usd` variant

The collision overlay above is the **static** ladder (fixed in place). A second script,
`scripts/omniverse/omniverse_ladder_rigid.py`, adds a **rigid-body
(carry-able)** variant using the exact same overlay trick — but sublayering the *collider*, not
the original:

```usda
#usda 1.0
( subLayers = [ @./<name>_collision.usd@ ] )   # pulls in the collider (which pulls in the original)

over "<defaultPrim>" ( prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"] )
{
    float physics:mass = 12.5      # size-based: round(clamp(2 + 3*height_m, 2, 40), 1)
}
```

Because it sublayers `<name>_collision.usd`, it inherits **all** the collision meshes for free,
then just applies `RigidBodyAPI` + a size-based `MassAPI` on the default prim. The file is only
~900 bytes — **no geometry is duplicated**. Mass scales with the ladder's height (taller ⇒
heavier, clamped 2–40 kg). CCD (helps thin rungs not tunnel) is enabled at the PhysX *scene*
level when the asset is used, so no Isaac-only `PhysxSchema` dependency is baked into the file.

## Three files per ladder — the scene builder picks one

| File | APIs added | Behaviour | Use |
|---|---|---|---|
| `<name>.usd` (original) | none | geometry only — robot clips through | visual prop ("nothing") |
| `<name>_collision.usd` | Collision + MeshCollision(convexDecomposition) + friction | **solid, fixed** — climbable, never moves | the ladder the G1 climbs |
| `<name>_collision_rigid.usd` | the above **+** RigidBody + Mass | **solid, movable** — climbable *and* can tip / be carried | a ladder to move or knock over |

Assembling a scene is just choosing which path to reference — no physics setup on the
scene-builder's side, it's all baked into the file. In an Isaac Lab asset config it's a
one-line `usd_path` swap.

**You select a tier by referencing it — you never delete files.** USD only loads the path you
reference plus its sublayers, and the dependency runs one way (downward):

```
_collision_rigid.usd ──► _collision.usd ──► <name>.usd
      (depends on)          (depends on)
```

- Reference `_collision.usd` → USD loads collision + original; the rigid file sits in the folder
  **ignored** (it costs ~900 bytes and has zero runtime effect unless referenced).
- Reference `_collision_rigid.usd` → USD loads all three.

Because the arrow points down, referencing the collider never pulls in the rigid overlay. So to
get a static ladder you just point at `_collision.usd` — no need to remove `_collision_rigid.usd`.

> **Keep the three files together.** `_collision_rigid.usd` sublayers `_collision.usd`, which sublayers the
> original — so the rigid file has *no geometry of its own*. Deleting the collision file breaks
> the rigid file. When copying/uploading, move all three as a set.

### How the tiers are linked — `subLayers`

The link between tiers is a **`subLayers`** entry in each file's header metadata — nothing more.
Each overlay declares, at the top of the file, that it stacks on top of the file below it:

```usda
# <name>_collision.usd
#usda 1.0
( subLayers = [ @./<name>.usd@ ] )            # stacks ON the original

# <name>_collision_rigid.usd
#usda 1.0
( subLayers = [ @./<name>_collision.usd@ ] )  # stacks ON the collision file
```

Reading the actual `subLayerPaths` of the three tiers:

| File | `subLayers` |
|---|---|
| `<name>.usd` (original) | `[]` — the base, links to nothing |
| `<name>_collision.usd` | `[ ./<name>.usd ]` |
| `<name>_collision_rigid.usd` | `[ ./<name>_collision.usd ]` |

Key facts this encodes:

- **Direction is downward.** The overlay points *down* to the file it builds on; the original
  references nothing. So the arrow is `rigid → collision → original`, never the reverse.
- **Composition, not copying.** Opening `_collision.usd` loads the sublayered original first (all
  its geometry, prims, materials), then composes this file's opinions (the `over` prims adding
  `CollisionAPI`) *on top* — the result is one merged scene. `_collision_rigid.usd` stacks its
  `RigidBodyAPI` + `Mass` opinions over that whole stack.
- **The path is `./` (same folder).** `_collision.usd` looks for `./<name>.usd` right beside it,
  so moving a file out of the folder breaks the link. This is the concrete reason the three USDs
  (plus the shared `Materials/` library the material paths point at) must travel together.

## Two formats handled in one script

| | Warehouse (DigitalTwin) | SimReady |
|---|---|---|
| Geometry | inline | behind instanceable references |
| Units | centimetres (`metersPerUnit=0.01`) | metres (`1.0`) |
| Default prim | `/World` | `/RootNode` |
| Extra step | — | de-instancing pass |

## In one sentence

For each mesh in a sublayered copy of the original, the script applies `CollisionAPI` +
`MeshCollisionAPI(convexDecomposition)` + a high-friction material, leaves it static, and saves
it as `<name>_collision.usd` — and PhysX cooks the actual convex pieces when Isaac loads it.
