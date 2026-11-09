"""
graph_generator.py

Builds the interaction graph used by Model 5 (Graph Attention Network +
Transformer).

Nodes = profiles (one node per profile_id)
Edges = messages or match requests between profiles
        edge attributes: {message_count, match_requested, timestamp}

FRAUD TOPOLOGIES injected (5, matching the Model 5 design doc):
  Star                 — ring hub contacts a shared victim pool directly
  Coordinated ring      — small cyclic groups of ring members messaging
                          each other, plus satellites converging on the
                          same victim pool as the hub
  Relay chain           — A -> B -> C -> D handoff across ring relays,
                          ending on a victim from the shared pool
  Demographic targeting — financial-scam profiles concentrate outreach on
                          a narrow age/income band instead of random legit
                          targets
  Isolation             — a subset of "other fraud" profiles contact only
                          very recently created (unverified-looking)
                          accounts

Legit traffic is demographic-aware (mild age/city/gender homophily) rather
than uniformly random, and includes realistic noise the model must not
confuse with fraud: some naturally popular high-in-degree profiles, and some
densely-interconnected friend-group clusters.

IMPORTANT — leakage: `is_fraud_edge` is NOT written to edges.csv. It (plus a
human-readable `topology` tag) is written only to a separate audit file,
`edges_audit.csv`, which must never be read by model feature code. `ring_id`
and `ring_role` are profile-level columns used here only to construct
topology; they are never written into edges.csv either.

The graph is serialized in three files:
  - edges.csv        (source, target, message_count, match_requested, timestamp)
  - graph_stats.csv   (per-node degree stats, for quick inspection / M5 node init)
  - edges_audit.csv   (source, target, is_fraud_edge, topology — audit only)

PyTorch Geometric loads from edges.csv in the Model 5 training script.
"""

from __future__ import annotations

import random

import networkx as nx
import numpy as np
import pandas as pd

NOISE_FLOOR = 0.05
CAP_BUFFER_DAYS = 45  # how far past the latest profile creation an edge may occur


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _factorize(series: pd.Series) -> np.ndarray:
    codes, _ = pd.factorize(series.fillna("__missing__"))
    return codes


def _clip_days(earliest_allowed: pd.Timestamp, desired_max_days: int, cap_date: pd.Timestamp) -> int:
    available = (cap_date - earliest_allowed).days
    if available <= 0:
        return 0
    return min(desired_max_days, available)


def _edge_timestamp(rng: np.random.Generator, created_a: pd.Timestamp, created_b: pd.Timestamp,
                     cap_date: pd.Timestamp, max_days: int, tight_hours: float | None = None) -> pd.Timestamp:
    """An edge can only occur after BOTH endpoints exist. `tight_hours` clusters
    coordinated (ring) edges tightly in time instead of spreading over months."""
    earliest = max(created_a, created_b)
    if tight_hours is not None:
        offset = pd.Timedelta(hours=float(rng.uniform(0.5, tight_hours)))
        ts = earliest + offset
        return min(ts, cap_date)
    days_cap = _clip_days(earliest, max_days, cap_date)
    offset_days = rng.integers(0, max(days_cap, 1) + 1)
    offset_minutes = int(rng.integers(0, 24 * 60))
    return earliest + pd.Timedelta(days=int(offset_days), minutes=offset_minutes)


def _weighted_target_positions(
    rng: np.random.Generator,
    pool_ages: np.ndarray, pool_gender_codes: np.ndarray,
    pool_city_codes: np.ndarray, pool_state_codes: np.ndarray,
    k: int, shortlist_size: int,
    source_age: float | None = None, source_gender_code: int | None = None,
    source_city_code: int | None = None, source_state_code: int | None = None,
    age_center_override: float | None = None, age_window: float = 15.0,
    w_age: float = 1.0, w_city: float = 1.0, w_gender: float = 1.0,
    pool_income: np.ndarray | None = None, w_income: float = 0.0,
    income_floor: float | None = None,
    eligible_mask: np.ndarray | None = None,
    exclude_position: int | None = None,
) -> np.ndarray:
    """Sample k positions (indices into the pool arrays) weighted by
    demographic compatibility. Returns an array of chosen positions
    (may be shorter than k if the eligible pool is small)."""
    n = len(pool_ages)
    eligible = np.arange(n) if eligible_mask is None else np.flatnonzero(eligible_mask)
    if exclude_position is not None:
        eligible = eligible[eligible != exclude_position]
    if len(eligible) == 0 or k <= 0:
        return np.array([], dtype=int)

    shortlist = rng.choice(eligible, size=min(shortlist_size, len(eligible)), replace=False)
    score = np.full(len(shortlist), NOISE_FLOOR, dtype=float)

    center_age = age_center_override if age_center_override is not None else source_age
    if center_age is not None:
        age_diff = np.abs(pool_ages[shortlist] - center_age)
        score = score + w_age * np.clip(1.0 - age_diff / age_window, 0.0, 1.0)

    if source_city_code is not None:
        city_match = (pool_city_codes[shortlist] == source_city_code).astype(float)
        state_match = (pool_state_codes[shortlist] == source_state_code).astype(float) * 0.4
        score = score + w_city * np.maximum(city_match, state_match)

    if source_gender_code is not None:
        score = score + w_gender * (pool_gender_codes[shortlist] != source_gender_code).astype(float)

    if pool_income is not None and w_income > 0 and income_floor is not None:
        income_score = np.clip(pool_income[shortlist] / max(income_floor, 1e-6), 0.0, 1.5)
        score = score + w_income * np.clip(income_score, 0.0, 1.0)

    score = score - score.min() + NOISE_FLOOR
    probs = score / score.sum()
    k = min(k, len(shortlist))
    return rng.choice(shortlist, size=k, replace=False, p=probs)


def build_interaction_graph(profiles_df: pd.DataFrame, seed: int | None = 42):
    """
    Given the full profiles dataframe, build a directed interaction graph.

    Returns:
        G         : networkx DiGraph
        edges_df  : model-safe edge table (no is_fraud_edge) for edges.csv
        stats_df  : per-node degree stats for graph_stats.csv
        audit_df  : source/target/is_fraud_edge/topology for edges_audit.csv
                    (NEVER feed this to model feature code)
    """
    rng = _rng(seed)
    random.seed(seed)

    df = profiles_df.reset_index(drop=True).copy()
    # format="ISO8601" is required: coordinated_ring profiles have
    # microsecond-precision created_at (from ring_created_at()'s fractional-
    # hour jitter) while every other profile has minute precision. Plain
    # pd.to_datetime() infers one format from the column and silently turns
    # every ring profile's timestamp into NaT otherwise.
    df["created_at_dt"] = pd.to_datetime(df["created_at"], format="ISO8601", errors="coerce")
    cap_date = df["created_at_dt"].max() + pd.Timedelta(days=CAP_BUFFER_DAYS)

    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_node(row["profile_id"], is_fraud=row["is_fraud"], fraud_type=row.get("fraud_type", None))

    is_fraud = df["is_fraud"].astype(bool)
    fraud_type = df.get("fraud_type", pd.Series([None] * len(df)))
    ring_id = df.get("ring_id", pd.Series([None] * len(df)))

    legit_df = df[~is_fraud].reset_index(drop=True)
    legit_ids = legit_df["profile_id"].to_numpy()
    legit_ages = legit_df["age"].to_numpy(dtype=float)
    legit_gender_codes = _factorize(legit_df["gender"])
    legit_city_codes = _factorize(legit_df["city"])
    legit_state_codes = _factorize(legit_df["state"])
    legit_income = legit_df["annual_income_lpa"].fillna(0.0).to_numpy(dtype=float)
    legit_created = legit_df["created_at_dt"].to_numpy()

    # ~45% of legit users are low-activity/dormant (new or rarely-logging-in
    # accounts): low outgoing contact count, and excluded as a target for
    # everyone else's outreach so they don't passively accumulate in-degree.
    # Calibrated off real matrimonial-platform engagement studies (Baihe:
    # ~half of users sent no initial message in their first 8 weeks; Match's
    # figure is directionally similar though weaker) rather than picked to
    # make the degree-only baseline look a particular way.
    low_activity_mask = rng.random(len(legit_ids)) < 0.45
    active_mask = ~low_activity_mask

    records = df.set_index("profile_id").to_dict(orient="index")

    edges: list[dict] = []
    audit: list[dict] = []

    def _add(source, target, message_count, match_requested, timestamp, is_fraud_edge, topology):
        edges.append({
            "source": source, "target": target,
            "message_count": message_count, "match_requested": match_requested,
            "timestamp": timestamp.isoformat(),
        })
        audit.append({"source": source, "target": target, "is_fraud_edge": is_fraud_edge, "topology": topology})

    # ── 1. Legitimate interaction edges — mildly demographic-aware ─────────
    # Baseline homophily (age/city/gender) so narrow targeting alone isn't a
    # fraud tell; fraud demographic-targeting (step 4 below) goes further.
    for pos, pid in enumerate(legit_ids):
        row = records[pid]
        if low_activity_mask[pos]:
            n_contacts = int(rng.integers(0, 6))
        else:
            n_contacts = min(int(row.get("unique_contacts", 5) or 5), 20)
        if n_contacts == 0:
            continue
        chosen = _weighted_target_positions(
            rng, legit_ages, legit_gender_codes, legit_city_codes, legit_state_codes,
            k=n_contacts, shortlist_size=250,
            source_age=legit_ages[pos], source_gender_code=legit_gender_codes[pos],
            source_city_code=legit_city_codes[pos], source_state_code=legit_state_codes[pos],
            age_window=8.0, w_age=1.0, w_city=0.8, w_gender=1.2,
            eligible_mask=active_mask, exclude_position=pos,
        )
        src_created = row["created_at_dt"]
        # Low-activity senders are excluded as a TARGET for everyone else
        # (that's the point — controls their degree), so a reply to their own
        # outgoing message is their only possible source of in-degree. At the
        # normal 50% reply rate that structurally yields an inherent ~2:1
        # out:in ratio for them — exactly the kind of directional asymmetry
        # that made satellites (bootstrapped from this same low-activity
        # pool) separable even once total degree overlapped. A near-1.0
        # reply rate keeps their own in/out roughly balanced instead.
        reply_chance = 0.92 if low_activity_mask[pos] else 0.5
        for tpos in chosen:
            tgt_pid = legit_ids[tpos]
            tgt_created = records[tgt_pid]["created_at_dt"]
            msg_count = int(rng.integers(1, 16))
            ts = _edge_timestamp(rng, src_created, tgt_created, cap_date, max_days=150)
            _add(pid, tgt_pid, msg_count, bool(rng.random() < 0.3), ts, False, "legit")
            if rng.random() < reply_chance:
                reply_ts = ts + pd.Timedelta(hours=float(rng.uniform(1, 72)))
                _add(tgt_pid, pid, int(rng.integers(1, msg_count + 1)), False, min(reply_ts, cap_date), False, "legit_reply")

    # ── 1b. Legit noise: popular high-in-degree profiles ───────────────────
    active_positions = np.flatnonzero(active_mask)
    n_popular = max(1, int(0.012 * len(legit_ids)))
    popular_positions = rng.choice(active_positions, size=min(n_popular, len(active_positions)), replace=False)
    for ppos in popular_positions:
        popular_pid = legit_ids[ppos]
        n_admirers = int(rng.integers(30, 81))
        admirer_positions = _weighted_target_positions(
            rng, legit_ages, legit_gender_codes, legit_city_codes, legit_state_codes,
            k=n_admirers, shortlist_size=300,
            source_age=legit_ages[ppos], source_gender_code=legit_gender_codes[ppos],
            source_city_code=legit_city_codes[ppos], source_state_code=legit_state_codes[ppos],
            age_window=10.0, w_age=0.8, w_city=0.6, w_gender=1.2,
            eligible_mask=active_mask, exclude_position=ppos,
        )
        popular_created = records[popular_pid]["created_at_dt"]
        for apos in admirer_positions:
            admirer_pid = legit_ids[apos]
            admirer_created = records[admirer_pid]["created_at_dt"]
            ts = _edge_timestamp(rng, popular_created, admirer_created, cap_date, max_days=150)
            _add(admirer_pid, popular_pid, int(rng.integers(1, 10)), bool(rng.random() < 0.4), ts, False, "legit_popular")
            if rng.random() < 0.6:
                reply_ts = ts + pd.Timedelta(hours=float(rng.uniform(1, 48)))
                _add(popular_pid, admirer_pid, int(rng.integers(1, 6)), False, min(reply_ts, cap_date), False, "legit_popular_reply")

    # ── 1c. Legit noise: dense friend-group clusters ────────────────────────
    n_clusters = max(1, int(len(legit_ids) / 1500))
    for _ in range(n_clusters):
        size = int(rng.integers(5, 16))
        if size > len(active_positions):
            continue
        member_positions = rng.choice(active_positions, size=size, replace=False)
        member_ids = legit_ids[member_positions]
        for i, a in enumerate(member_ids):
            for b in member_ids:
                if a == b or rng.random() > 0.55:
                    continue
                a_created = records[a]["created_at_dt"]
                b_created = records[b]["created_at_dt"]
                ts = _edge_timestamp(rng, a_created, b_created, cap_date, max_days=150)
                _add(a, b, int(rng.integers(1, 12)), bool(rng.random() < 0.2), ts, False, "legit_cluster")

    # ── Bootstrap pool: each fraud profile below draws its in/out-degree
    # TARGETS from an actual legit (in_degree, out_degree) pair realized so
    # far, rather than from a hand-picked random range. Matching the joint
    # distribution (not just a mean) is what actually closes the degree_ratio
    # gap — tuning in_degree and out_degree ranges independently kept leaving
    # a directional (out-heavy) signature a degree-only classifier could
    # still exploit even when total degree overlapped.
    legit_id_to_pos = {pid: i for i, pid in enumerate(legit_ids)}
    legit_out_count = np.zeros(len(legit_ids), dtype=int)
    legit_in_count = np.zeros(len(legit_ids), dtype=int)
    for e in edges:
        s_pos = legit_id_to_pos.get(e["source"])
        if s_pos is not None:
            legit_out_count[s_pos] += 1
        t_pos = legit_id_to_pos.get(e["target"])
        if t_pos is not None:
            legit_in_count[t_pos] += 1
    low_activity_positions = np.flatnonzero(low_activity_mask)

    def _bootstrap_target(pool_positions: np.ndarray, jitter: tuple[float, float] = (0.85, 1.15)) -> tuple[int, int]:
        bp = pool_positions[rng.integers(0, len(pool_positions))]
        target_out = max(0, int(round(legit_out_count[bp] * rng.uniform(*jitter))))
        target_in = max(0, int(round(legit_in_count[bp] * rng.uniform(*jitter))))
        return target_out, target_in

    # ── 2-3. Coordinated ring: star + relay chain + satellite convergence + cycles
    ring_mask = (fraud_type == "coordinated_ring")
    if ring_mask.any() and ring_id.notna().any():
        for _, ring_group in df[ring_mask].groupby(ring_id):
            hub_ids = ring_group.loc[ring_group["ring_role"] == "hub", "profile_id"].tolist()
            relay_ids = ring_group.loc[ring_group["ring_role"] == "relay", "profile_id"].tolist()
            satellite_ids = ring_group.loc[ring_group["ring_role"] == "satellite", "profile_id"].tolist()
            all_ring_ids = hub_ids + relay_ids + satellite_ids
            if not all_ring_ids:
                continue

            pool_size = int(rng.integers(150, 351))
            pool_positions = rng.choice(len(legit_ids), size=min(pool_size, len(legit_ids)), replace=False)
            shared_victim_pool = legit_ids[pool_positions]

            # Star: each hub contacts most of the shared pool directly.
            for hub_id in hub_ids:
                n_victims = int(rng.integers(100, len(shared_victim_pool) + 1)) if len(shared_victim_pool) > 100 else len(shared_victim_pool)
                victims = rng.choice(shared_victim_pool, size=min(n_victims, len(shared_victim_pool)), replace=False)
                hub_created = records[hub_id]["created_at_dt"]
                for v in victims:
                    ts = _edge_timestamp(rng, hub_created, records[v]["created_at_dt"], cap_date, max_days=0, tight_hours=72)
                    _add(hub_id, v, int(rng.integers(1, 6)), True, ts, True, "ring_star")
                    if rng.random() < 0.05:
                        reply_ts = ts + pd.Timedelta(hours=float(rng.uniform(1, 24)))
                        _add(v, hub_id, 1, False, min(reply_ts, cap_date), False, "ring_star_reply")

            # Relay chain: handoff across relays, final hop into the shared
            # pool. `while i < n` (not `i + 2 < n`) guarantees every relay
            # ends up in some chain — a tail-end remainder of 1-2 relays is
            # folded into its own short chain instead of being dropped, which
            # previously left stranded, completely edgeless relays.
            shuffled_relays = list(relay_ids)
            random.shuffle(shuffled_relays)
            i = 0
            n_relays = len(shuffled_relays)
            while i < n_relays:
                remaining = n_relays - i
                if remaining < 3:
                    chain_len = remaining
                else:
                    chain_len = min(random.randint(3, 5), remaining)
                    if remaining - chain_len == 1:
                        chain_len = remaining  # avoid stranding exactly 1 relay
                chain = shuffled_relays[i:i + chain_len]
                for j in range(len(chain) - 1):
                    a, b = chain[j], chain[j + 1]
                    ts = _edge_timestamp(rng, records[a]["created_at_dt"], records[b]["created_at_dt"], cap_date, max_days=0, tight_hours=72)
                    _add(a, b, int(rng.integers(5, 21)), False, ts, True, "ring_relay")
                if chain and len(shared_victim_pool):
                    victim = shared_victim_pool[rng.integers(0, len(shared_victim_pool))]
                    ts = _edge_timestamp(rng, records[chain[-1]]["created_at_dt"], records[victim]["created_at_dt"], cap_date, max_days=0, tight_hours=72)
                    _add(chain[-1], victim, int(rng.integers(1, 6)), True, ts, True, "ring_relay")
                i += chain_len

            # Satellites: low, varied degree, converging on the shared pool
            # and on each other (not isolated nodes anymore). The in/out
            # target is bootstrapped from the legit low-activity segment's
            # actual degree pairs, so satellites sit inside that segment's
            # joint (in, out) shape rather than being a directionally
            # distinct (out-heavy) population a degree-only model can spot.
            for sat_id in satellite_ids:
                target_out, target_in = _bootstrap_target(low_activity_positions, jitter=(0.8, 1.6))
                if target_out == 0 and target_in == 0:
                    # Avoid reintroducing a zero-degree node, but symmetrically
                    # — flooring only target_out (as an earlier version of
                    # this did) systematically biases satellites toward
                    # out-heavy degree_ratio, since the low-activity bootstrap
                    # pool hits 0 often on either side.
                    target_out = 1
                sat_created = records[sat_id]["created_at_dt"]
                n_internal = int(round(target_out * rng.uniform(0.10, 0.30)))
                n_victims = max(0, target_out - n_internal)
                if n_victims and len(shared_victim_pool):
                    victims = rng.choice(shared_victim_pool, size=min(n_victims, len(shared_victim_pool)), replace=False)
                    for v in victims:
                        ts = _edge_timestamp(rng, sat_created, records[v]["created_at_dt"], cap_date, max_days=0, tight_hours=96)
                        _add(sat_id, v, int(rng.integers(1, 4)), True, ts, True, "ring_satellite_convergence")
                if n_internal:
                    other_members = [m for m in all_ring_ids if m != sat_id]
                    if other_members:
                        targets = random.sample(other_members, k=min(n_internal, len(other_members)))
                        for t in targets:
                            ts = _edge_timestamp(rng, sat_created, records[t]["created_at_dt"], cap_date, max_days=0, tight_hours=96)
                            _add(sat_id, t, int(rng.integers(1, 4)), False, ts, True, "ring_satellite_convergence")
                if target_in:
                    sender_positions = _weighted_target_positions(
                        rng, legit_ages, legit_gender_codes, legit_city_codes, legit_state_codes,
                        k=target_in, shortlist_size=150,
                        w_age=0.3, w_city=0.3, w_gender=0.3,
                        eligible_mask=active_mask,
                    )
                    for spos in sender_positions:
                        sender_pid = legit_ids[spos]
                        ts = _edge_timestamp(rng, records[sender_pid]["created_at_dt"], sat_created, cap_date, max_days=90)
                        _add(sender_pid, sat_id, int(rng.integers(1, 5)), bool(rng.random() < 0.3), ts, True, "ring_satellite_inbound")

            # Small cycles among ring members — "a small group cycles
            # messages among themselves", distinct from star/relay/chain.
            if len(all_ring_ids) >= 3:
                n_cycles = max(1, len(all_ring_ids) // 20)
                for _ in range(n_cycles):
                    group = random.sample(all_ring_ids, k=min(3, len(all_ring_ids)))
                    for j in range(len(group)):
                        a, b = group[j], group[(j + 1) % len(group)]
                        ts = _edge_timestamp(rng, records[a]["created_at_dt"], records[b]["created_at_dt"], cap_date, max_days=0, tight_hours=48)
                        _add(a, b, int(rng.integers(2, 10)), False, ts, True, "ring_cycle")

    # ── 4. Demographic targeting: financial scam concentrates on a narrow,
    #      valuable age/income band rather than uniform-random legit targets.
    financial_df = df[fraud_type == "financial_scam"]
    for _, frow in financial_df.iterrows():
        fid = frow["profile_id"]
        n_targets = int(frow.get("unique_contacts", 50) or 50)
        target_age_center = float(rng.uniform(34, 55))
        income_floor = float(np.quantile(legit_income, 0.60))
        chosen = _weighted_target_positions(
            rng, legit_ages, legit_gender_codes, legit_city_codes, legit_state_codes,
            k=n_targets, shortlist_size=350,
            age_center_override=target_age_center, age_window=6.0,
            w_age=1.6, w_city=0.3, w_gender=0.6,
            pool_income=legit_income, w_income=1.4, income_floor=income_floor,
        )
        f_created = records[fid]["created_at_dt"]
        for tpos in chosen:
            tgt_pid = legit_ids[tpos]
            ts = _edge_timestamp(rng, f_created, records[tgt_pid]["created_at_dt"], cap_date, max_days=60)
            _add(fid, tgt_pid, int(rng.integers(5, 26)), True, ts, True, "demographic_targeting")
            if rng.random() < 0.03:
                reply_ts = ts + pd.Timedelta(hours=float(rng.uniform(2, 96)))
                _add(tgt_pid, fid, int(rng.integers(1, 4)), False, min(reply_ts, cap_date), False, "demographic_targeting_reply")

    # ── 5. Other fraud: half isolation pattern (new accounts only), half
    #      generic sparse outreach (unchanged baseline behavior).
    other_mask = is_fraud & ~fraud_type.isin(["coordinated_ring", "financial_scam"])
    other_df = df[other_mask]
    other_ids = other_df["profile_id"].tolist()
    random.shuffle(other_ids)
    split_point = len(other_ids) // 2
    isolation_ids = set(other_ids[:split_point])

    all_legit_positions = np.arange(len(legit_ids))
    for _, orow in other_df.iterrows():
        oid = orow["profile_id"]
        # Bootstrap this profile's in/out-degree targets from an actual legit
        # (in_degree, out_degree) pair rather than unique_contacts — matches
        # legit's joint degree shape (including ratio), not just its mean.
        n_targets, n_inbound = _bootstrap_target(all_legit_positions)
        n_targets = max(1, n_targets)
        o_created = records[oid]["created_at_dt"]
        if oid in isolation_ids:
            reference_time = o_created + pd.Timedelta(days=float(rng.uniform(5, 60)))
            recency_mask = np.abs((legit_created - np.datetime64(reference_time)).astype("timedelta64[D]").astype(float)) <= 20
            # Near the end of the 6-month window there may be no legit
            # accounts left within +/-20 days of reference_time (nothing was
            # created after the simulation's cutoff) — fall back to the
            # most-recently-created legit accounts available as the closest
            # proxy for "new users" rather than leaving the profile edgeless.
            if int(recency_mask.sum()) < max(n_targets, 5):
                fallback_n = max(50, n_targets * 5)
                most_recent_positions = np.argsort(legit_created)[-fallback_n:]
                recency_mask = np.zeros(len(legit_created), dtype=bool)
                recency_mask[most_recent_positions] = True
            # ~65% new accounts / ~35% older accounts — a softer, less
            # mechanically-perfect isolation signature than contacting new
            # accounts exclusively.
            n_new = int(round(n_targets * 0.65))
            n_old = n_targets - n_new
            chosen_new = _weighted_target_positions(
                rng, legit_ages, legit_gender_codes, legit_city_codes, legit_state_codes,
                k=n_new, shortlist_size=300,
                w_age=0.0, w_city=0.0, w_gender=0.0,
                eligible_mask=recency_mask,
            )
            chosen_old = _weighted_target_positions(
                rng, legit_ages, legit_gender_codes, legit_city_codes, legit_state_codes,
                k=n_old, shortlist_size=250,
                w_age=0.3, w_city=0.3, w_gender=0.3,
            )
            chosen = np.concatenate([chosen_new, chosen_old])
            topology = "isolation_pattern"
        else:
            chosen = _weighted_target_positions(
                rng, legit_ages, legit_gender_codes, legit_city_codes, legit_state_codes,
                k=n_targets, shortlist_size=250,
                w_age=0.3, w_city=0.3, w_gender=0.3,
            )
            topology = "other_fraud_generic"
        # max_days=150 and a ~50% reply chance match the legit-legit loop's
        # own defaults exactly (section 1 above) — these profiles' edges
        # should otherwise look like ordinary legit traffic in timing and
        # reply rate; a single-feature check found response_rate alone was
        # AUC~0.96 separable before this fix, purely because these edges
        # never got a reciprocal reply while legit ones do ~50-92% of the
        # time — a generator artifact, not a real fraud signal.
        for tpos in chosen:
            tgt_pid = legit_ids[tpos]
            tgt_created = records[tgt_pid]["created_at_dt"]
            msg_count = int(rng.integers(1, 11))
            ts = _edge_timestamp(rng, o_created, tgt_created, cap_date, max_days=150)
            _add(oid, tgt_pid, msg_count, bool(rng.random() < 0.4), ts, True, topology)
            if rng.random() < 0.5:
                reply_ts = ts + pd.Timedelta(hours=float(rng.uniform(1, 72)))
                _add(tgt_pid, oid, int(rng.integers(1, msg_count + 1)), False, min(reply_ts, cap_date), False, f"{topology}_reply")

        # Degree-overlap fix: these fraud profiles would otherwise have
        # near-zero in-degree (only the rare organic reply), sitting well
        # below the legit distribution. A handful of legit users messaging
        # them first (browsing, mild interest before red flags show) is
        # realistic and pulls their total degree into legit's range.
        # n_inbound is the bootstrapped target_in from above.
        sender_positions = _weighted_target_positions(
            rng, legit_ages, legit_gender_codes, legit_city_codes, legit_state_codes,
            k=n_inbound, shortlist_size=250,
            w_age=0.3, w_city=0.3, w_gender=0.3,
            eligible_mask=active_mask,
        )
        for spos in sender_positions:
            sender_pid = legit_ids[spos]
            sender_created = records[sender_pid]["created_at_dt"]
            msg_count = int(rng.integers(1, 8))
            ts = _edge_timestamp(rng, sender_created, o_created, cap_date, max_days=150)
            _add(sender_pid, oid, msg_count, bool(rng.random() < 0.3), ts, True, "legit_to_fraud_inbound")
            if rng.random() < 0.5:
                reply_ts = ts + pd.Timedelta(hours=float(rng.uniform(1, 72)))
                _add(oid, sender_pid, int(rng.integers(1, msg_count + 1)), False, min(reply_ts, cap_date), False, "legit_to_fraud_inbound_reply")

    # Add all edges to the graph
    for e in edges:
        G.add_edge(e["source"], e["target"], message_count=e["message_count"],
                   match_requested=e["match_requested"], timestamp=e["timestamp"])

    edges_df = pd.DataFrame(edges, columns=["source", "target", "message_count", "match_requested", "timestamp"])
    audit_df = pd.DataFrame(audit, columns=["source", "target", "is_fraud_edge", "topology"])

    # ── Per-node graph statistics (useful features for M5 node init) ──────
    stats = []
    for node in G.nodes():
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        in_w = sum(d.get("message_count", 1) for _, _, d in G.in_edges(node, data=True))
        out_w = sum(d.get("message_count", 1) for _, _, d in G.out_edges(node, data=True))
        stats.append({
            "profile_id": node,
            "in_degree": in_deg,
            "out_degree": out_deg,
            "in_msg_weight": in_w,
            "out_msg_weight": out_w,
            "degree_ratio": out_deg / max(in_deg, 1),
            "response_rate": in_deg / max(out_deg, 1),
            "is_fraud": G.nodes[node].get("is_fraud", False),
        })
    stats_df = pd.DataFrame(stats)

    return G, edges_df, stats_df, audit_df
