# Omniverse Asset-Pack Scan Log

We're working through NVIDIA's free [Omniverse USD asset packs](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html#d-openusd-asset-packs)
looking for things the Fiatlux benchmark needs: **light bulbs, lamps, light fixtures,
sockets, and ladders**.

**Standard procedure for every pack:**
1. Download the ZIP (`scripts/omniverse/omniverse_pack_download.py`) to a
   local staging dir (transient — deleted after curation).
2. Scan it (`scripts/omniverse/omniverse_pack_scan.py`) and write the
   **full detailed inventory** here — every category → subgroup (count) → each item, one per bullet.
3. Extract the Fiatlux-relevant assets (lamps, bulbs, fixtures, ladders, climb structures), author
   physics + restore materials, and upload the **curated set to GCS** —
   `gs://fiatlux/assets/omniverse_{ladder,climb,lamp,bulb}/`.
4. The ZIP and local staging are transient; the **canonical assets live in GCS** (synced via
   [`assets/download_assets.sh`](../assets/download_assets.sh)). The destination annotations below
   are GCS paths.

A few words used below:
- **design** = one named asset (e.g. "TiltAndRoll ladder").
- **model files** = the individual `.usd` files for a design (one per size/color/config). One design usually has several.

---

## Progress

| # | Pack | Size | What we kept | Bulbs/Lamps? |
|---|------|------|--------------|--------------|
| 1 | Warehouse | 18 GB | **86 ladder models** + **Mezzanine/OfficeSet climb structures (141)** | none |
| 2 | Residential | 22.9 GB | **30 lighting/fixture models + 1 ladder** | ✓ lamps, but no separable bulb |
| 3 | SimReady Warehouse 01 | 13.9 GB | **4 work platforms + 7 ladders** | none |
| 4 | SimReady Furniture & Misc | 9.4 GB | ❌ nothing relevant | none |
| 5 | Commercial | 5.8 GB | ❌ nothing relevant | none |
| 6 | Industrial | 1.8 GB | ❌ nothing relevant | none |
| 7 | Showcase Scenes | 9.2 GB | ❌ nothing relevant | none |
| 8 | Sample Scenes | 26.7 GB | **LightBulb (separable!) + OldAttic lamps** | ✓ **real separable bulb** |
| 9 | Default Scene Templates | 24 MB | ❌ nothing relevant | none |
| 10 | SimReady Warehouse 02 | 20.9 GB | ❌ nothing relevant | none |
| 11 | SimReady Containers & Shipping 01 | 21.9 GB | ❌ nothing relevant | none |
| 12 | SimReady Containers & Shipping 02 | 21.1 GB | ❌ nothing relevant | none |
| 13 | Data Center | 9.8 GB | ❌ nothing relevant | none |
| 14 | Rigged Characters | 935 MB | ❌ nothing relevant | none |

Packs skipped entirely (not asset catalogs): materials, skies, automotive, city/AEC.
All 14 prop/scene packs are scanned below — even the ones with nothing for us.

Disk: scanned ZIPs and local staging are transient; the curated assets live in GCS
(`gs://fiatlux/assets/omniverse_*/`).

---

## Pack 1 — Warehouse  (18 GB, 1,441 model files)

A warehouse prop pack — **1,441 model files** (material/texture files excluded).
Full breakdown, every category → subgroup (count) → items:

#### Storage — 284
- **Containers (121)**
  - Container_F: 27
  - Container_C: 24
  - Container_H: 22
  - Container_B: 18
  - Container_D: 9
  - Container_E: 6
  - Container_G: 6
  - Container_A: 4
  - Container_I: 4
  - Container_J: 1
- **Totes (56)**
  - Tote_C: 21
  - Tote_D: 14
  - Tote_B: 9
  - Tote_E: 6
  - Tote_A: 4
  - Tote_F: 2
- **Drums (21)**
  - Fiber_A: 7
  - Plastic_A: 5
  - Steel_A: 4
  - Plastic_B: 3
  - NestablePlastic_A: 2
- **Cases (18)**
  - Case_A: 5
  - Case_C: 5
  - Case_D: 4
  - Case_E: 3
  - Case_B: 1
- **Jugs (16)**
  - Plastic_Jerrican_A: 5
  - F-Style_A: 4
  - Milk_A: 3
  - Utility_A: 3
  - Metal_F-Style_A: 1
- **Bins (15)**
  - Bin_B: 7
  - Bin_A: 6
  - Bins: 2
- **Pails (13)**
  - Plastic_A: 4
  - ScrewTop_A: 4
  - LiquidScrewTop_A: 2
  - Square_A: 2
  - ClosedHead_A: 1
- **Boxes (12)**
  - Box_A: 11
  - Boxes: 1
- **Bottles (7)**
  - WhitePacker_A: 4
  - NaturalBostonRound_A: 3
- **IBC_Tanks (5)**
  - Tank_A: 2
  - StainlessSteel_A: 1
  - Tank_B: 1
  - Tank_Heater_A: 1

#### Shipping — 282
- **Cardboard_Boxes (96)**
  - White_A: 36
  - Flat_A: 16
  - Long_A: 12
  - Multi-Depth_A: 12
  - Cube_A: 10
  - Printer_A: 10
- **Pallets (65)**
  - Drum_A: 9
  - Block_A: 9
  - Pressed_A: 9
  - Recycled_A: 8
  - Export_A: 7
  - SolidTopRackable_A: 4
  - Economy_A: 3
  - Block_B: 3
  - Aluminum_A: 2
  - HeavyDutyNestable_A: 2
  - Rackable_A: 2
  - GalvanizedSteel_A: 1
  - ClosedDeckExport_A: 1
  - IBCSpillContainment_A: 1
  - Industrial_A: 1
  - RackableExport_A: 1
  - Block_C: 1
  - Wing_A: 1
- **Wood_Crates (57)**
  - Standard_A: 29
  - PalletCollar_A: 10
  - HeavyDuty_A: 6
  - Plywood_A: 6
  - Plywood_B: 6
- **Wood_Crate_on_Pallet (51)**
  - Standard_A: 29
  - PalletCollar_A: 10
  - HeavyDuty_A: 6
  - Plywood_A: 6
- **Cardboard_Boxes_on_Pallet (13)**
  - Pallet_Asm_A: 6
  - Pallet_Asm_B: 2
  - Pallet_Asm_C: 2
  - Pallet_Asm_D: 2
  - Cardboard_Boxes_on_Pallet: 1

#### Modular — 275
- **Warehouse_A (134)**
  - .SubUSDs: 85
  - Warehouse_A: 49
- **OfficeSet_A (102)**
  - OfficeSet_A: 63
  - .SubUSDs: 39
- **Mezzanine_A (39)**
  - Mezzanine_A: 38
  - .SubUSDs: 1

#### Shelving — 251
- **Racks (146)**
  - Rack_K: 24
  - Rack_H: 23
  - Rack_L: 18
  - Rack_C: 16
  - Rack_M: 13
  - Rack_A: 12
  - Rack_F: 12
  - Rack_G: 12
  - Rack_B: 6
  - Rack_N: 6
  - Rack_D: 2
  - Rack_E: 2
- **Industrial (101)**
  - IndustrialSteel_A: 23
  - HeavyDutySteel_A: 18
  - OpenSteel_A: 17
  - WireShelving_A: 12
  - TireRackSystem_C: 9
  - TireRackSystem_B: 5
  - BulkStorageRack_A: 4
  - TireRackSystem_A: 4
  - TireRackSystem_D: 4
  - MobileStorageRack_A: 3
  - TireRack_A: 2
- **Bar_and_Sheet (4)**
  - Bar_and_Sheet: 4

#### Equipment — 204
- **Ladders (86)**
  - AlumStep_A: 10
  - FRP_Extention_A: 8
  - AlumMultiPurpose_A: 6
  - AlumStepDouble_A: 6
  - AlumStep_D: 6
  - AlumStep_C: 6
  - FoldingStep_A: 6
  - FRP_Step_A: 6
  - AlumStep_B: 5
  - GripStepRolling_A: 5
  - NarrowAisleRolling_A: 5
  - TiltAndRoll_A: 4
  - Workplatform_A: 4
  - AlumStepStand_A: 3
  - HeavyDutyFRPStep_A: 3
  - RollingScaffold_A: 3
- **Conveyors (62)**
  - ConveyorBelt_A: 61
  - ConveyorAsm_A: 1
- **Ramps (34)**
  - DockBoard_A: 8
  - DockBoard_B: 7
  - AluminiumWalkRamp_A: 6
  - DockBoard_C: 6
  - WheelRiserRamp_A: 4
  - OptionalRamp_A: 1
  - PolyCurbRamp_A: 1
  - WorkRamp_A: 1
- **Hand_Trucks (12)**
  - Convertible_Aluminum_A: 4
  - Std_Alum_A: 3
  - Gas_Cylinder_A: 2
  - Convertible_Steel_A: 2
  - Keg_A: 1
- **Drum_Handling (4)**
  - Truck_A: 2
  - Cradle_A: 1
  - DispensingTruck_A: 1
- **Forklifts (3)**
  - Forklift_A: 1
  - Forklift_B: 1
  - Forklift_C: 1
- **Pallet_Trucks (3)**
  - Heavy_Duty_A: 1
  - Low_Profile_A: 1
  - Scale_A: 1

#### Safety — 96
- **Tape (60)**
  - Safety_A: 6
  - Safety_B: 6
  - Safety_C: 6
  - VinylMessage_A: 6
  - VinylMessage_B: 6
  - VinylMessage_C: 6
  - VinylMessage_D: 6
  - VinylMessage_E: 6
  - Reflective_I: 6
  - Reflective_J: 6
- **Cones (23)**
  - Traffic: 9
  - Heavy-Duty_Traffic: 7
  - Pop-Up_Traffic: 5
  - Colored_Sport: 1
  - Colored_Traffic: 1
- **Floor_Signs (13)**
  - Warning_A: 4
  - MobileShutOffSystem_A: 3
  - WetFloor_A: 2
  - WetFloor_B: 2
  - FoldingWarning_A: 1
  - Movable_A: 1

#### Furnishing — 35
- **Cabinets (35)**
  - Cabinet_A: 10
  - ClosedIndSteel_A: 8
  - SteelCompartmentWithFeet_A: 7
  - Cabinet_B: 4
  - Ventilated_A: 3
  - WeldedParts_A: 2
  - Luggage_A: 1

#### Facilities — 8
- **Cleaning (8)**
  - Utility_A: 2
  - Bucket_A: 1
  - MopBucket_A: 1
  - MopBucket_B: 1
  - Trolley_A: 1
  - Trolley_B: 1
  - Trolley_C: 1

#### Electronics — 6
- **Barcode (6)**
  - Equipment_C: 1
  - Scanner_A: 1
  - Scanner_B: 1
  - Scanner_C: 1
  - Scanner_D: 1
  - Scanner_E: 1

> **🧗 Climb-relevant finding in Modular:** the **Mezzanine_A** kit is made of elevated
> 2-floor platforms with **8 built-in ladder pieces** connecting the floors, and
> **OfficeSet_A** adds **9 more ladder pieces** + mezzanine decks. These are full
> *climb-to-a-platform* scenarios (ladder + elevated floor), not just standalone ladders
> — potentially a ready-made environment for the G1 climbing task.

### What we kept: the ladders
**16 ladder designs, 86 model files (902 MB)** → `gs://fiatlux/assets/omniverse_ladder/ (Warehouse designs)`.
This includes ladder types we couldn't find before (extension, A-frame, scaffold):

| Ladder design | Model files | What it is |
|---|---|---|
| FRP_Extention_A | 8 | **fiberglass extension ladder** (tall, vertical — the rung type we lacked) |
| AlumStepDouble_A | 6 | **A-frame, double-sided** step ladder |
| AlumMultiPurpose_A | 6 | foldable multi-purpose ladder |
| RollingScaffold_A | 3 | **rolling scaffold** |
| Workplatform_A | 4 | work platform |
| AlumStep_A / B / C / D | 10 / 5 / 6 / 6 | aluminum step ladders |
| FRP_Step_A | 6 | fiberglass step ladder |
| HeavyDutyFRPStep_A | 3 | heavy-duty fiberglass step |
| FoldingStep_A | 6 | folding step stool |
| GripStepRolling_A | 5 | rolling step ladder |
| NarrowAisleRolling_A | 5 | narrow-aisle rolling ladder |
| TiltAndRoll_A | 4 | tilt-and-roll platform (same family as our original 7) |
| AlumStepStand_A | 3 | step stand (same family as our original 7) |

### 💡 Bulbs / lamps: none
It's a warehouse pack — the only lighting is ceiling fixtures baked into the office /
mezzanine building kits, not standalone lamps. (Lamps should come from Residential.)

### Not kept yet (available if we want them later)
Ramps (34), shelving/racks (251), steel cabinets (35) for scene dressing, 14 ready-made
PhysX physics-material presets, and — most interesting — the **Mezzanine_A / OfficeSet_A
climb structures** (ladder + elevated platform).

---

## Pack 2 — Residential  (22.9 GB, ~507 models)

A home-interior pack — **507 model files**. Full breakdown, every category → subgroup (count) → items:

#### Decor — 192
- Sculptures: 38
- Books: 35
- Pictures: 31
- Vases: 28
- Tchotchkes: 11
- Kitchenware: 8
- Statues: 8
- Clocks: 7
- CandleHolders: 6
- Magazines: 4
- WallSculptures: 3
- DeskDecor: 2
- Mirrors: 2
- PowerOutlets: 2
- Rugs: 2
- Shelves: 2
- Coasters: 1
- Globes: 1
- WindowDressing: 1

#### Furniture — 114
- **FurnitureSets (27)**
  - Appleseed: 10
  - Whitecliff: 5
  - Crestwood: 3
  - Dutchtown: 3
  - Hazelwood: 3
  - Maplewood: 3
- **Outdoors (22)**
  - Roxana: 12
  - Dellwood: 10
- Bookshelves: 15
- **Sofas (14)**
  - Lemay: 12
  - Sofas: 2
- Chairs: 9
- **DiningSets (8)**
  - EastRural: 3
  - Jennings: 3
  - DesPeres: 2
- MediaTables: 5
- SofaTables: 3
- BarStools: 2
- CoffeeTables: 2
- EndTables: 2
- Beds: 1
- Desks: 1
- Misc: 1
- Partitions: 1
- Tables: 1

#### Kitchen — 57
- **Kitchenware (51)**
  - KitchenUtensils: 19
  - Cookware: 8
  - Dinnerware: 5
  - Flatware: 5
  - Serving: 5
  - StorageAndOrganization: 4
  - GadgetsAndTools: 3
  - Cutlery: 1
  - CuttingBoards: 1
- KitchenFaucets: 3
- Dispensers: 2
- Griddles: 1

#### Entertainment — 40
- **Games (25)**
  - Games: 10
  - ChessSet: 8
  - DiceSet: 7
- GameRoom: 14
- Toys: 1

#### Outdoors — 32
- Planters: 19
- Lighting: 7
- Decor: 4
- FirePits: 2

#### Lighting — 22
- Table Lamps: 13
- Floor Lamps: 5
- Candles: 2
- Chandeliers: 2

#### Food — 19
- Fruit: 6
- Candy: 5
- Boxed: 4
- Containers: 2
- Berries: 1
- Vegetables: 1

#### Misc — 9
- Supplies: 3
- PaintTray: 2
- Dropcloths: 1
- Ladders: 1
- PaintRoller: 1
- Pavers: 1

#### Appliances — 7
- Oven: 2
- Dishwashers: 1
- Fans: 1
- Microwaves: 1
- Refrigerators: 1
- Vents: 1

#### Electronics — 7
- GPU: 2
- Nvidia_Shield: 2
- Cameras: 1
- Speakers: 1
- Televisions: 1

#### Plants — 6
- Plants: 6

#### Fixtures — 2
- Fixtures: 2

### What we kept: lighting + fixtures + ladder
**27 model files + 1 ladder (1.2 GB)** → `gs://fiatlux/assets/omniverse_lamp/ (+ omniverse_ladder for the 1 ladder)`:

| Asset | Model files |
|---|---|
| Table Lamps | 13 |
| Floor Lamps | 5 |
| Outdoor lighting | 7 |
| Chandeliers | 2 |
| Ladder (`Misc/Ladders`) | 1 |

(The 2 ceiling/wall fixtures found in this pack were not uploaded — lamp total = 27.)

### ⚠️ Important: probably no *separable* bulb
There is **no standalone light-bulb asset** in this pack, and these ArchVis lamps are
most likely **one fused model** (base + shade + bulb together), not a removable bulb in a
socket. The Fiatlux task needs a bulb you can take out and replace — which at the time of this
scan pointed at the BEHAVIOR-1K lamps (with socket metadata). That is no longer the case: the
shipped pair is the separable Omniverse LightBulb from pack 8; the BEHAVIOR-1K lamps are no
longer loaded by any task, and survive only as opt-in scene dressing (see `assets/README.md`). So treat these as **fixture / lamp
variety and scene dressing** unless inspection shows the bulb is a separate part.

---

## Pack 3 — SimReady Warehouse 01  (13.9 GB, 139 distinct props)

NVIDIA **SimReady** format (each asset ships 4 USD files: main + `_base` + `_inst` +
`_inst_base`; counts below are distinct props, not files). All under
`Assets/simready_content/common_assets/props/`. This is the catalog our **original 7
ladders** came from.

#### Shelving & racks — 99
- industrialsteelshelving: 23
- tireracksystem: 22
- openindustrialsteelshelving: 18
- sm_heavydutysteelshelving: 13
- wireshelving: 12
- heavydutysteelshelving: 5
- bulkstoragerack: 4
- tirerack: 2

#### Dock boards & ramps — 17
- dockboard: 15
- ramplong: 1
- rampshort: 1

#### Pallets & fencing — 11
- recycledwoodpallet: 8
- metalfencing: 3

#### 🪜 Ladders — 7  *(kept)*  → `gs://fiatlux/assets/omniverse_ladder/ (SimReady ladders)` (990 MB)
- tiltandrollladder: 4  (a01–a04)
- aluminumstepstand: 3  (a01–a03)
- (these are the same 7 as our original Phase-1 set)

#### Work platforms — 4  *(kept)*  → `gs://fiatlux/assets/omniverse_ladder/ (SimReady work platforms)` (860 MB)
- stationaryworkplatform: a01, a02, a03, a04

#### Misc — 1
- industrial: 1

---

## Pack 4 — SimReady Furniture & Misc  (9.4 GB, 202 distinct props)

SimReady format. Despite the name, **no lamps / lights / fixtures / ladders** — the only
"light" in the whole pack is a `light_rig.usd` thumbnail-render helper (not an asset).
Full itemized breakdown (every prop family with count; SimReady = 4 files/asset, counts are distinct props):

**Safety — tape & cones (78)**
- deluxesafetytape: 17
- trafficcone: 8
- heavydutytrafficcone: 6
- vinylmessagetape8m75mm: 5
- vinylmessagetape6m75mm: 5
- vinylmessagetape12m75mm: 5
- vinylmessagetape2m75mm: 5
- vinylmessagetape10m75mm: 5
- vinylmessagetape4m75mm: 5
- popupcone: 3
- reflectivetape8m5cm: 1
- reflectivetape8m10cm: 1
- reflectivetape10m5cm: 1
- reflectivetape4m5cm: 1
- coloredsporttrafficcone: 1
- coloredtrafficcone: 1
- reflectivetape4m10cm: 1
- reflectivetape12m10cm: 1
- reflectivetape12m5cm: 1
- reflectivetape6m10cm: 1
- reflectivetape2m10cm: 1
- reflectivetape2m5cm: 1
- reflectivetape6m5cm: 1
- reflectivetape10m10cm: 1

**Kitchenware (31)**
- spatula: 19
- kettle: 1
- plate_small: 1
- spoon_small: 1
- plate_large: 1
- blackandbrassbowl_small: 1
- cutting_board_a: 1
- fork_small: 1
- kettle_seat: 1
- blackandbrassbowl_large: 1
- serving_bowl: 1
- spoon_big: 1
- fork_big: 1

**Food (13)**
- orange: 2
- lemon: 2
- lychee01: 1
- redonion: 1
- steam: 1
- pumpkinlarge: 1
- lime01: 1
- pomegranate01: 1
- avocado01: 1
- peach: 1
- pumpkinsmall: 1

**Seating (24)**
- dellwood_diningchair: 1
- crestwood_sofa: 1
- crestwood_loveseat: 1
- desperes_chair: 1
- chair_array: 1
- petite_chair: 1
- willowbench: 1
- waiting: 1
- crestwood_chair: 1
- dellwood_ottoman: 1
- petite_loveseat: 1
- armchair: 1
- salon: 1
- zag_corner: 1
- waitingbench: 1
- sappington_chair: 1
- zag_backless: 1
- appleseed_sofatable: 1
- bar_stool: 1
- zag_middle: 1
- sedie_chair: 1
- chair_wire: 1
- birch_lowbackseat: 1
- birch_highbackseat: 1

**Tables & desks (15)**
- oaktablelarge: 1
- capricetable_yin: 1
- capricetable_b: 1
- l_desk: 1
- dellwood_endtable: 1
- desk: 1
- contemporary_lowfulldesk: 1
- dellwood_coffeetable: 1
- capricetable_c: 1
- appleseed_coffeetable: 1
- contemporary_lowhalfdesk: 1
- appleseed_endtable: 1
- oaktablesmall: 1
- dellwood_rounddiningtable: 1
- dellwood_diningtable: 1

**Storage & shelving (13)**
- cabinet: 4
- standard_smallunit: 1
- contemporary_lockedstorage: 1
- contemporary_openstorage: 1
- contemporary_storagecube: 1
- cornerrail: 1
- standard_largeunit: 1
- standard_halfunit: 1
- cornershelf_square: 1
- contemporary_laterialfile: 1

**Decor (7)**
- seenoevil_skull: 1
- hearnoevil_skull: 1
- crabbypenholder: 1
- monarch: 1
- antiquelvase: 1
- speaknoevil_skull: 1
- antiquelvasesmall: 1

**Planters (3)**
- gardenplanter_small: 1
- gardenplanter_medium: 1
- gardenplanter_large: 1

**Ramp (1)**
- optionalramp: 1

**Other / uncategorized (17)**
- anza_large: 1
- anza_medium: 1
- costello: 1
- whitehome01: 1
- contemporary_risefull: 1
- contemporary_riseshort: 1
- rave: 1
- cline: 1
- modern: 1
- gilbert: 1
- dentist: 1
- danny: 1
- boat: 1
- modularcurved: 1
- seahorn: 1
- curved_full: 1
- roland: 1

---

## Pack 5 — Commercial  (5.8 GB, 82 models)

Office furniture (ArchVis/Commercial). **No lamps / lights / fixtures / ladders.**

#### Seating — 40
- Seating: 26
- Caprice: 8
- Jobba: 3
- Zag: 3

#### Storage — 19
- Contemporary: 9
- Standard: 6
- Storage: 4

#### Tables — 11
- Tables: 7
- Caprice: 4

#### Conference — 6
- Conference: 6

#### Reception — 6
- Reception: 6

---

## Pack 6 — Industrial  (1.8 GB, 75 models)

Industrial ArchVis props. **No lamps / lights / fixtures / ladders.**

#### Racks — 27
- Racks: 27

#### Containers — 18
- Cardboard: 12
- Wooden: 6

#### Piles — 12
- Piles: 12

#### Railing — 6
- Railing: 6

#### Shelves — 6
- Shelves: 6

#### Pallets — 3
- Pallets: 3

#### Buildings — 2
- Warehouse: 2

#### Stages — 1
- Stages: 1

---

## Pack 7 — Showcase Scenes  (9.2 GB, 158 USD — demo scenes)

Assembled **demo scenes** (not a prop catalog), in two versions (2023_1, 2023_2_1):

#### 2023_1/IsaacWarehouse — 64
- Racks: 18
- Cardboard: 12
- Piles: 12
- Wooden: 6
- Railing: 6
- Shelves: 4
- Pallets: 3
- Warehouse: 2
- IsaacWarehouse: 1

#### 2023_1/Ragnarok — 11
- SubUSDs: 10
- Ragnarok: 1

#### 2023_2_1/ConceptCar — 8
- SubUSDs: 7
- ConceptCar: 1

#### 2023_2_1/IsaacWarehouse — 64
- Racks: 18
- Cardboard: 12
- Piles: 12
- Wooden: 6
- Railing: 6
- Shelves: 4
- Pallets: 3
- Warehouse: 2
- IsaacWarehouse: 1

#### 2023_2_1/Ragnarok — 11
- SubUSDs: 10
- Ragnarok: 1

---

## Pack 8 — Sample Scenes  (26.7 GB, 939 USD — demo/rendering scenes)

Assembled showcase scenes (each also mirrored under `Examples/Rendering/`). Scenes:

- **Marbles** 282 — physics marble-run demo
- **OldAttic** 109 — attic room (← has the lamps/string-lights)
- **Automotive_Material_Library** 24 (+12 pristine) — car material demo
- **Flight** 11 · **EuclidVR** 4 · **Claire** 2 · **Astronaut** 1
- **Visual Scripting** 1 — (← contains the LightBulb)

### KEPT — lighting (71 MB) → `gs://fiatlux/assets/omniverse_bulb/ (LightBulb only; other SampleScenes lamps not uploaded)`
**🔌 LightBulb** (`Visual Scripting/LightBulb`) — the ONLY standalone, *modeled* light
bulb in any Omniverse pack. Geometry is structured as a desk fixture with a **separable
bulb**: `/World/Geom/BulbGrp` = **Bulb** (glass) + **Base** (screw cap), distinct from the
lamp `/World/Geom/Base` (feet + NVIDIA logo). → the bulb can be removed from the fixture,
which is exactly what the Fiatlux bulb-swap task needs (the ArchVis lamps were fused).

**OldAttic/Props** also holds lamps (`TableLamp`, `table_light`, 5× `string_lights_strand`,
5× `StringLightLights` string-light bulbs), but these were **not uploaded** — the lamp set is
sourced from the Residential pack (`omniverse_lamp`, 27 models), so the attic props are redundant.

---

## Pack 9 — Default Scene Templates  (24 MB, 16 USD)

10 empty starter-room template scenes (Basic ×5, Default, Interior, LookDev ×2, Outdoor).
**No lamps / bulbs / ladders / fixtures.**

---

## Pack 10 — SimReady Warehouse 02  (20.9 GB, 178 distinct props)

SimReady format. **All storage racks + pallets + ramps — no ladders, platforms, or lamps.**

- **Racks (148):** rack 132, sm_rack 13, horizontalbarrack 3, verticalbarrack 1 (+ `solidtoprackablepallet` etc.)
- **Pallets (~25):** aluminumpallet, rackablepallet, heavydutynestablepallet, closeddeckexportpallet, rackableexportpallet, + ~13 numbered `pallet_asm_*`
- **Ramps (5):** wheelriserramp 4, workramp 1 (same as Warehouse/WH01 — already noted)

---

## Pack 11 — SimReady Containers & Shipping 01  (21.9 GB, 316 props)

SimReady format, all storage/shipping vessels. **No ladders / platforms / lamps / bulbs.**

- **Containers 106, Totes 55, Bins 12** · **Boxes 107** (whitecorrugated 36, flat 16, multidepth 12, long 12, card 11, printers 10, cube 10) · **Drums 11** (fiber 7, steel 4) · **Bottles/Jugs/Buckets 9**

---

## Pack 12 — SimReady Containers & Shipping 02  (21.1 GB, 194 props)

SimReady format, crates/pallets/drums. **No ladders / platforms / lamps / bulbs.**

- **Wood crates (75):** standardwoodcrate 28, stdwoodcrateasm 29, heavyduty/plywood variants
- **Pallets (48):** blockpallet 13, woodpalletcollar(asm) 20, exportpallet 7, wooddrumpallet 8
- **Drums (16):** plasticdrum 8, + others · **Pails/Jugs/Bottles (22)** · **IBC tanks 3** · **Boxes 11**

---

## Pack 13 — Data Center  (9.8 GB, 75 models)

Data-center equipment (DigitalTwin/Datacenter). **No lamps / bulbs / ladders / fixtures.**

- **Network_Switches (24)**: SubUSDs 11, Common 5, QM8700 3, AS4600 1, SN2700 1, SN3700 1, SN4600 1, UA950H-2SF 1
- **Facilities (18)**: Data_Hall 7, SubUSDs 7, Cable_Tray 4
- **Racks (9)**: Accessories 5, Fiber_A 1, SubUSDs 1, .SubUSDs 1, Rack_42U_A 1
- **Server_Nodes (5)**: Servers 3, .SubUSDs 1, DGX 1
- **Liquid_Cooling (3)**: SubUSDs 2, DCP_A 1
- **Power_Distribution (2)**: PDU_A 1, SubUSDs 1

---

## Pack 14 — Rigged Characters  (935 MB, 116 USD)

Reallusion **rigged human characters**: Debra, Orc, Worker, + ActorCore people
(Business_F, Party_M, Uniform_F/M) with motions. **No lamps / bulbs / ladders.**
(The one 'light_fixture' hit is Debra's stage prop — a false positive.)
