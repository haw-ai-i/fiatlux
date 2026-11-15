Changelog
---------

0.2.0 (2026-11-14)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* S05 and S07's arrival gates are debounced over ``ARRIVAL_SUSTAIN_SECONDS`` like S09's, instead
  of scoring on the single frame where every conjunct happens to line up (issue #267, the fix
  #206 made on S09 and these two never inherited).
* All three arrival gates (S05, S07, S09) now require the success predicate to have been observed
  FALSE before it can fire, via ``mdp.sustained(require_transition=True)``. A robot spawned inside
  the success region holds every conjunct indefinitely, so a debounce window alone still scored it
  (issue #266).
* The Replace layout sampler rejects draws that leave the ladder-mount spawn within
  ``MIN_CARRY_CLEARANCE`` of the disposal zone. Non-overlapping zones were not enough to give a
  carry leg a carry: measured over 200 seeds, 2 spawned inside S05's own success region, the worst
  0.695 m from the crate (issue #266).

Changed
^^^^^^^

* **A few seeds now draw a different Replace layout.** Only a REJECTED draw consumes extra rng;
  a seed whose first draw already cleared ``MIN_CARRY_CLEARANCE`` is bit-identical to 0.1.0.
  Measured over 60 seeds, 3 changed (5%): seeds 5, 40 and 43. Seed 5 is one of the two degenerate
  layouts this release fixes, so of the benchmark's first ten seeds the only one that moved is the
  one that was scoring a carry leg without a carry. Scores for those seeds are not comparable
  across 0.1.0 and 0.2.0; the rest are unaffected.
* ``LADDER_MOUNT_RADIUS`` is now defined in ``scene_cfg`` and re-exported from ``mdp.nav_terms``;
  the sampler needs it and ``scene_cfg`` cannot import from ``mdp``.

0.1.0 (2026-02-13)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Created an initial template for building an extension or project based on Isaac Lab