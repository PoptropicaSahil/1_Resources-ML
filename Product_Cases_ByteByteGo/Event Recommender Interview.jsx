import { useState } from "react";

const Section = ({ title, time, phase, children, defaultOpen = false }) => {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="mb-6 border border-gray-200 rounded-xl overflow-hidden bg-white">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-6 py-4 bg-gray-50 hover:bg-gray-100 transition-colors text-left"
      >
        <div className="flex items-center gap-3">
          <span className="text-xs font-mono bg-indigo-100 text-indigo-700 px-2 py-1 rounded">
            {phase}
          </span>
          <span className="font-semibold text-gray-900 text-lg">{title}</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-sm text-gray-500">{time}</span>
          <span className="text-gray-400 text-xl">{open ? "−" : "+"}</span>
        </div>
      </button>
      {open && <div className="px-6 py-5">{children}</div>}
    </div>
  );
};

const Speaker = ({ role, children }) => {
  const isInterviewer = role === "interviewer";
  return (
    <div className={`mb-4 flex gap-3 ${isInterviewer ? "" : ""}`}>
      <div
        className={`shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold ${
          isInterviewer
            ? "bg-orange-100 text-orange-700"
            : "bg-blue-100 text-blue-700"
        }`}
      >
        {isInterviewer ? "I" : "C"}
      </div>
      <div className="flex-1">
        <div
          className={`text-xs font-semibold mb-1 ${
            isInterviewer ? "text-orange-700" : "text-blue-700"
          }`}
        >
          {isInterviewer ? "Interviewer" : "Candidate"}
        </div>
        <div className="text-gray-800 leading-relaxed text-sm">{children}</div>
      </div>
    </div>
  );
};

const MathBlock = ({ children }) => (
  <div className="bg-gray-50 border border-gray-200 rounded-lg px-4 py-3 my-3 font-mono text-sm overflow-x-auto">
    {children}
  </div>
);

const Callout = ({ type = "insight", title, children }) => {
  const styles = {
    insight: "bg-purple-50 border-purple-200 text-purple-900",
    tradeoff: "bg-amber-50 border-amber-200 text-amber-900",
    mental: "bg-emerald-50 border-emerald-200 text-emerald-900",
    warning: "bg-red-50 border-red-200 text-red-900",
  };
  const icons = {
    insight: "💡",
    tradeoff: "⚖️",
    mental: "🧠",
    warning: "⚠️",
  };
  return (
    <div
      className={`my-4 border rounded-lg px-4 py-3 text-sm ${styles[type]}`}
    >
      <div className="font-semibold mb-1">
        {icons[type]} {title}
      </div>
      {children}
    </div>
  );
};

const Diagram = ({ title, children }) => (
  <div className="my-5 bg-gray-900 rounded-xl p-5 overflow-x-auto">
    {title && (
      <div className="text-gray-400 text-xs font-mono mb-3 uppercase tracking-wider">
        {title}
      </div>
    )}
    <div className="text-gray-100 font-mono text-xs leading-relaxed whitespace-pre">
      {children}
    </div>
  </div>
);

const MetricTable = ({ headers, rows }) => (
  <div className="my-4 overflow-x-auto">
    <table className="w-full text-sm border-collapse">
      <thead>
        <tr>
          {headers.map((h, i) => (
            <th
              key={i}
              className="bg-gray-100 text-left px-3 py-2 border border-gray-200 font-semibold text-gray-700"
            >
              {h}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, i) => (
          <tr key={i}>
            {row.map((cell, j) => (
              <td key={j} className="px-3 py-2 border border-gray-200">
                {cell}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

export default function EventRecommenderInterview() {
  return (
    <div className="max-w-4xl mx-auto p-4 sm:p-8 bg-gray-100 min-h-screen">
      {/* Header */}
      <div className="mb-8 text-center">
        <div className="text-xs font-mono text-gray-500 mb-2 tracking-wider">
          ML SYSTEM DESIGN INTERVIEW — SENIOR DATA SCIENTIST (L5/L6)
        </div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Design an Event Recommender System
        </h1>
        <p className="text-gray-500 text-sm">
          45–60 min • Full math & intuition • Multi-stage funnel architecture
        </p>
        <div className="flex justify-center gap-4 mt-4 text-xs text-gray-500">
          <span className="bg-orange-100 text-orange-700 px-2 py-1 rounded">
            I = Interviewer
          </span>
          <span className="bg-blue-100 text-blue-700 px-2 py-1 rounded">
            C = Candidate (10 YoE)
          </span>
        </div>
      </div>

      {/* Interview timeline */}
      <div className="mb-6 bg-white rounded-xl p-4 border border-gray-200">
        <div className="text-xs font-semibold text-gray-500 mb-3 uppercase tracking-wider">
          Interview Timeline
        </div>
        <div className="flex flex-wrap gap-2 text-xs">
          {[
            ["Clarify", "5-8m", "bg-red-100 text-red-700"],
            ["Metrics", "5m", "bg-orange-100 text-orange-700"],
            ["Architecture", "8m", "bg-yellow-100 text-yellow-800"],
            ["Features", "8-10m", "bg-green-100 text-green-700"],
            ["Models", "12-15m", "bg-blue-100 text-blue-700"],
            ["Serving", "5-8m", "bg-indigo-100 text-indigo-700"],
            ["Monitor", "5m", "bg-purple-100 text-purple-700"],
          ].map(([label, time, cls]) => (
            <span key={label} className={`px-3 py-1.5 rounded-full font-medium ${cls}`}>
              {label} ({time})
            </span>
          ))}
        </div>
      </div>

      {/* ═══════════ PHASE 1: REQUIREMENTS ═══════════ */}
      <Section
        title="Requirements Clarification"
        time="5–8 min"
        phase="01"
        defaultOpen={true}
      >
        <Speaker role="interviewer">
          Let's say you're at a company like Eventbrite or Meetup. You need to
          design an ML system that recommends events to users. Walk me through
          how you'd approach this.
        </Speaker>

        <Speaker role="candidate">
          Before diving in, I'd like to clarify a few things to scope the
          problem properly.
          <br />
          <br />
          <strong>Product surface:</strong> Are we recommending events on a
          homepage feed, in email digests, or as search results? I'll assume a{" "}
          <strong>homepage personalized feed</strong> — the highest-traffic
          surface.
          <br />
          <br />
          <strong>Event types:</strong> Are these in-person events (concerts,
          meetups, conferences), virtual, or both? I'll assume{" "}
          <strong>primarily in-person with some virtual</strong>.
          <br />
          <br />
          <strong>Scale:</strong> What order of magnitude are we talking?
        </Speaker>

        <Speaker role="interviewer">
          Let's say 50M monthly active users, 500K active events at any time,
          concentrated in 200 metro areas globally. Primarily a homepage feed.
        </Speaker>

        <Speaker role="candidate">
          Great. Let me also clarify a few domain-specific constraints that make
          events fundamentally different from, say, movie or product
          recommendations:
          <br />
          <br />
          <strong>1. Temporal perishability:</strong> Events expire. A concert on
          March 5th is useless on March 6th. Our system must handle inventory
          that's constantly churning — roughly 10-20K new events/day, 10-20K
          expiring.
          <br />
          <br />
          <strong>2. Geographic constraint:</strong> Unlike Netflix or Spotify,
          physical proximity is a hard filter. A user in San Francisco won't
          attend a meetup in Tokyo (usually).
          <br />
          <br />
          <strong>3. Severe cold-start:</strong> Most events are new and have
          zero interaction history. Unlike products that live for months, the
          average event might only exist for 2-4 weeks before occurring. We
          can't rely heavily on collaborative filtering alone.
          <br />
          <br />
          <strong>4. Capacity constraints:</strong> Events have limited seats.
          Once sold out, recommending them is harmful.
          <br />
          <br />
          <strong>5. Implicit vs explicit signals:</strong> Users rarely "rate"
          events. Our primary signals are clicks, RSVPs, ticket purchases, and
          attendance.
        </Speaker>

        <Speaker role="interviewer">
          Those are exactly the right constraints to surface. What assumptions
          will you make?
        </Speaker>

        <Speaker role="candidate">
          <strong>Assumptions:</strong>
          <br />• User is logged in (we have user history)
          <br />• We have location data (GPS or user-set city)
          <br />• Events have metadata: title, description, category, venue,
          time, price, organizer
          <br />• Latency budget: &lt;200ms p99 for the full recommendation
          pipeline
          <br />• We need to generate ~50 recommendations per page load
          <br />• We serve ~1000 requests/sec at peak
        </Speaker>

        <Callout type="mental" title="Mental Model — Events vs. Static Items">
          The key distinction for event recommendation vs. general RecSys is the{" "}
          <strong>item lifecycle</strong>. In product recommendations, item
          embeddings can be precomputed and cached for weeks. In event
          recommendation, the item corpus is a <em>sliding window</em> — you're
          constantly indexing new items and expiring old ones. This fundamentally
          affects your ANN index refresh strategy and cold-start approach.
        </Callout>
      </Section>

      {/* ═══════════ PHASE 2: METRICS ═══════════ */}
      <Section title="Metrics Definition" time="5 min" phase="02">
        <Speaker role="interviewer">
          How would you measure success for this system?
        </Speaker>

        <Speaker role="candidate">
          I'd structure metrics at three levels: business, online, and offline.
          <br />
          <br />
          <strong>Business Metrics</strong> — what the CEO cares about:
          <br />• <em>Ticket revenue / RSVP volume</em> per user per month
          <br />• <em>Event discovery rate</em> — % of events getting ≥N
          RSVPs from recommendations (supply-side health)
          <br />• <em>User retention</em> — DAU/MAU ratio
          <br />
          <br />
          <strong>Online Metrics</strong> — what we A/B test on:
          <br />• <em>Click-through rate (CTR)</em> on recommended events
          <br />• <em>RSVP rate</em> — stronger signal than clicks
          <br />• <em>Conversion rate</em> — click → RSVP → actual attendance
          <br />• <em>Recommendation diversity</em> — entropy across event
          categories in top-10
          <br />
          <br />
          <strong>Offline Metrics</strong> — for model iteration:
        </Speaker>

        <MetricTable
          headers={["Metric", "Formula / Intuition", "Used For"]}
          rows={[
            [
              "Recall@K",
              "Of all relevant events, what fraction did we retrieve in top K?",
              "Retrieval stage",
            ],
            [
              "NDCG@K",
              "Are the most relevant events ranked highest? Penalizes relevant items ranked low.",
              "Ranking stage",
            ],
            [
              "MAP@K",
              "Average precision across all users — rewards correct ordering",
              "End-to-end",
            ],
            [
              "AUC-ROC",
              "Probability that a positive (RSVP'd) event is scored higher than a negative",
              "Pointwise ranker",
            ],
            [
              "Log Loss",
              "−[y·log(p) + (1−y)·log(1−p)] — calibration of probability estimates",
              "CTR prediction model",
            ],
          ]}
        />

        <Speaker role="candidate">
          One critical nuance: <strong>CTR alone is a trap</strong> for events.
          A clickbait-y event title gets clicks but not RSVPs. So I'd use a{" "}
          <strong>composite objective</strong>:
        </Speaker>

        <MathBlock>
          score(user, event) = w₁·P(click) + w₂·P(RSVP|click) +
          w₃·P(attend|RSVP)
        </MathBlock>

        <Speaker role="candidate">
          Where w₁ &lt; w₂ &lt; w₃ to weight deeper funnel actions more heavily.
          Concretely, in practice I might use w₁=0.1, w₂=0.3, w₃=0.6. We'd
          tune these weights via online A/B tests watching the business metrics.
        </Speaker>

        <Callout type="mental" title="Mental Model — Metric Hierarchy">
          Think of metrics as a pyramid: <em>offline metrics</em> give fast
          iteration signal (minutes), <em>online metrics</em> validate via A/B
          tests (days), and <em>business metrics</em> confirm long-term impact
          (weeks). If your offline metrics improve but online metrics don't, your
          offline evaluation setup is broken (e.g., data leakage, selection
          bias). If online metrics improve but business metrics don't, your
          proxy objective is misaligned with true user value.
        </Callout>

        <Speaker role="interviewer">
          How do you handle the position bias problem in offline evaluation?
        </Speaker>

        <Speaker role="candidate">
          Great callout. Position bias is the fact that users click items at position 1
          more than position 5 regardless of relevance. For offline evaluation, I'd use{" "}
          <strong>Inverse Propensity Scoring (IPS)</strong>:
        </Speaker>

        <MathBlock>
          IPS-weighted reward = Σᵢ (rᵢ / P(examine position i)){"\n"}
          where P(examine position i) is estimated from position-click curves in
          logs
        </MathBlock>

        <Speaker role="candidate">
          We can estimate examination probabilities by running a randomization
          experiment — swap items at positions i and j and measure click
          differences. The ratio gives you the position bias curve.
        </Speaker>
      </Section>

      {/* ═══════════ PHASE 3: ARCHITECTURE ═══════════ */}
      <Section title="High-Level Architecture" time="8 min" phase="03">
        <Speaker role="interviewer">
          Walk me through the system architecture end to end.
        </Speaker>

        <Speaker role="candidate">
          I'll use the standard <strong>multi-stage funnel</strong> pattern — the
          same approach Instagram Explore, YouTube, and Pinterest use. The key
          insight is that each stage trades off between recall and precision,
          with computational cost increasing as the funnel narrows.
        </Speaker>

        <Diagram title="Multi-Stage Recommendation Funnel">
          {`┌─────────────────────────────────────────────────────────┐
│                    EVENT CORPUS                          │
│                   500K active events                     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              STAGE 0: GEO + BUSINESS FILTER             │
│  Hard filters: location radius, sold-out, date range    │
│  500K → ~20K candidates         Latency: <5ms           │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│         STAGE 1: CANDIDATE RETRIEVAL (Recall)           │
│  Two-Tower model + ANN search (HNSW/ScaNN)              │
│  Multiple retrieval channels merged                     │
│  20K → ~500 candidates          Latency: <20ms          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│        STAGE 2: PRE-RANKING (Light Scoring)             │
│  Lightweight model (distilled from ranker)              │
│  500 → ~100 candidates          Latency: <15ms          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│          STAGE 3: RANKING (Precision)                   │
│  Deep model: DCN-v2 / MTML with cross-features         │
│  Multi-task: P(click), P(RSVP), P(attend)               │
│  100 → ~50 scored items         Latency: <50ms          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│          STAGE 4: RE-RANKING (Business Logic)           │
│  Diversity injection, freshness boost, fairness         │
│  Organizer exposure guarantees, dedup                   │
│  50 → 50 re-ordered             Latency: <10ms          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
                   [ Final 50 events ]`}
        </Diagram>

        <Speaker role="candidate">
          <strong>Why this architecture?</strong>
          <br />
          <br />
          Scoring all 500K events with the heavy ranking model would take ~500K ×
          0.5ms = 250 seconds per request. That's obviously impossible with a
          200ms budget. The funnel lets us spend our compute budget wisely:
          cheap models see many items, expensive models see few.
          <br />
          <br />
          The <strong>latency budget decomposition</strong> is:
        </Speaker>

        <MetricTable
          headers={["Stage", "Latency", "Items", "Compute/Item"]}
          rows={[
            ["Geo + Business Filter", "~5ms", "500K → 20K", "O(1) lookups"],
            ["Retrieval (Two-Tower + ANN)", "~20ms", "20K → 500", "ANN query"],
            ["Pre-Ranking", "~15ms", "500 → 100", "~0.03ms/item"],
            ["Ranking (Deep Model)", "~50ms", "100 → 50", "~0.5ms/item"],
            ["Re-Ranking", "~10ms", "50 → 50", "Rule-based"],
            ["Network/Orchestration", "~30ms", "—", "—"],
            ["TOTAL", "~130ms p50", "—", "< 200ms p99"],
          ]}
        />

        <Callout type="tradeoff" title="Tradeoff — Pre-Ranking: Is It Worth It?">
          Pre-ranking adds latency and complexity. You can skip it if your
          retrieval is precise enough (say, returns ~100 candidates). But at our
          scale (500 candidates from retrieval), scoring 500 items with a heavy
          DCN model at 0.5ms/item = 250ms — that blows our budget. The
          pre-ranker acts as <em>knowledge distillation at serving time</em>:
          it's a lightweight student model trained to approximate the full
          ranker's output, cutting 5x candidates at 10x speed.
        </Callout>

        <Diagram title="Offline/Online System Split">
          {`
┌───── OFFLINE (Batch/Streaming) ─────┐    ┌───── ONLINE (Real-time) ──────────┐
│                                      │    │                                   │
│  ┌──────────────┐  ┌──────────────┐  │    │  Request: (user_id, location,     │
│  │ Training Data │  │ Feature      │  │    │           timestamp)              │
│  │ Generation    │  │ Engineering  │  │    │         │                         │
│  │ (click/RSVP   │  │ Pipeline     │  │    │         ▼                         │
│  │  logs)        │  │              │  │    │  ┌─────────────┐                  │
│  └──────┬───────┘  └──────┬───────┘  │    │  │ Feature     │ ← Feature Store  │
│         │                 │          │    │  │ Fetching    │   (Redis/Feast)   │
│         ▼                 ▼          │    │  └──────┬──────┘                   │
│  ┌──────────────┐  ┌──────────────┐  │    │         │                         │
│  │ Model        │  │ Feature      │  │    │         ▼                         │
│  │ Training     │  │ Store        │──│────│  ┌─────────────┐                  │
│  │ (GPU cluster)│  │ (Offline)    │  │    │  │ Multi-Stage │                  │
│  └──────┬───────┘  └──────────────┘  │    │  │ Funnel      │                  │
│         │                            │    │  └──────┬──────┘                   │
│         ▼                            │    │         │                         │
│  ┌──────────────┐  ┌──────────────┐  │    │         ▼                         │
│  │ Event Index  │  │ Model        │  │    │  Response: ranked event list      │
│  │ (ANN rebuild │  │ Registry     │──│────│                                   │
│  │  every ~1hr) │  │              │  │    │                                   │
│  └──────────────┘  └──────────────┘  │    │                                   │
└──────────────────────────────────────┘    └───────────────────────────────────┘`}
        </Diagram>

        <Speaker role="candidate">
          One important detail for events specifically: the ANN index needs to be{" "}
          <strong>refreshed frequently</strong> — at least hourly — because 
          events are constantly being created and expiring. Compare this to a product
          catalog where you might rebuild the index daily. I'd use an{" "}
          <strong>incremental ANN index</strong> (like Milvus or Vespa) that
          supports real-time inserts and deletes, rather than a batch-rebuilt
          FAISS index.
        </Speaker>
      </Section>

      {/* ═══════════ PHASE 4: FEATURES ═══════════ */}
      <Section title="Feature Engineering" time="8–10 min" phase="04">
        <Speaker role="interviewer">
          What features would you use, and how do you handle the cold-start
          problem?
        </Speaker>

        <Speaker role="candidate">
          I'll organize features by entity and signal freshness. This matters
          because different features have different update cadences and serving
          costs.
        </Speaker>

        <MetricTable
          headers={["Category", "Features", "Update Freq", "Storage"]}
          rows={[
            [
              "User — Static",
              "age_bucket, gender, city, account_age, preferred_categories (from profile)",
              "Daily",
              "Feature Store",
            ],
            [
              "User — Behavioral",
              "past_RSVPs_by_category (sparse vector), avg_event_price, avg_distance_traveled, time_of_day_preference, recency_weighted_interaction_embedding",
              "Hourly",
              "Feature Store",
            ],
            [
              "User — Real-time",
              "current_location, current_session_clicks, time_since_last_visit",
              "Per-request",
              "Computed online",
            ],
            [
              "Event — Content",
              "title_embedding (BERT), category, sub_category, price_bucket, is_free, is_virtual, duration_hours, venue_embedding",
              "At creation",
              "Feature Store",
            ],
            [
              "Event — Popularity",
              "total_RSVPs, RSVP_velocity (RSVPs/hour in last 24h), page_views, organizer_avg_rating, seats_remaining_pct",
              "Every 15min",
              "Feature Store",
            ],
            [
              "Context — Temporal",
              "day_of_week, hour_of_day, is_weekend, days_until_event, is_holiday_week",
              "Per-request",
              "Computed online",
            ],
            [
              "Cross — User×Event",
              "user_organizer_affinity, user_category_affinity, user_venue_distance_km, user_price_preference_match, social_signal (friends attending)",
              "Per-request (ranking only)",
              "Computed online",
            ],
          ]}
        />

        <Speaker role="interviewer">
          How do you generate the text embeddings for event titles and descriptions?
        </Speaker>

        <Speaker role="candidate">
          For the <strong>retrieval stage</strong>, I'd use a distilled sentence
          transformer (e.g., all-MiniLM-L6-v2 or E5-small) to generate
          384-dimensional embeddings. These are precomputed offline when an event is
          created.
          <br />
          <br />
          For the <strong>ranking stage</strong>, I don't pass raw BERT embeddings.
          Instead, I'd extract a few semantic features: the top-3 predicted
          categories from a text classifier, sentiment score, and keyword
          features. The ranking model learns its own feature interactions —
          giving it raw 384-dim embeddings would be wasteful.
          <br />
          <br />
          Now, for <strong>cold-start</strong> — this is the core challenge for events:
        </Speaker>

        <Diagram title="Cold-Start Strategy by User/Event Matrix">
          {`
                        Known Event          New Event (Cold)
                    ┌──────────────────┬──────────────────────┐
                    │                  │                      │
   Known User       │  Collaborative   │  Content-based       │
                    │  filtering +     │  (text embeddings +  │
                    │  behavioral      │  category match +    │
                    │  features        │  organizer affinity) │
                    │                  │                      │
                    ├──────────────────┼──────────────────────┤
                    │                  │                      │
   New User (Cold)  │  Popularity +    │  Global popularity + │
                    │  geo-contextual  │  trending events +   │
                    │  (trending in    │  onboarding quiz     │
                    │   your city)     │  preferences         │
                    │                  │                      │
                    └──────────────────┴──────────────────────┘`}
        </Diagram>

        <Speaker role="candidate">
          The two-tower model is particularly good for cold-start because the
          event tower can produce an embedding from{" "}
          <strong>content features alone</strong> — title, category, price, venue
          location — without needing any interaction history. The moment an event
          is created, it gets an embedding and enters the ANN index. This is a
          major advantage over pure collaborative filtering approaches like
          matrix factorization, where a new item with zero interactions has no
          representation.
          <br />
          <br />
          For <strong>new users</strong>, I'd use a two-phase approach: (1) show
          popular/trending events in their geo for the first session, (2) after
          they interact with 3-5 events, switch to the personalized model. We
          can also use an <strong>onboarding preference quiz</strong> (like
          Spotify Wrapped categories) to bootstrap the user embedding.
        </Speaker>

        <Callout type="mental" title="Mental Model — Feature Serving Tiers">
          Think of features in three serving tiers by latency cost:
          <br />
          <strong>Tier 1 (precomputed, &lt;1ms):</strong> Looked up from feature
          store by key. User and event static features.
          <br />
          <strong>Tier 2 (near-real-time, &lt;5ms):</strong> Aggregated from
          streaming pipeline (Kafka → Flink). Event popularity, RSVP velocity.
          <br />
          <strong>Tier 3 (computed at request time, &lt;10ms):</strong>{" "}
          Cross-features like distance, social overlap. These can only exist in
          the ranking stage where you have both user and event context.
          <br />
          <br />
          The two-tower retrieval model can <em>only</em> use Tier 1 features
          (because user and event towers must be independent for caching). The
          ranking model uses all three tiers — that's why it's more powerful.
        </Callout>
      </Section>

      {/* ═══════════ PHASE 5: MODEL DESIGN ═══════════ */}
      <Section title="Model Design & Training" time="12–15 min" phase="05">
        <Speaker role="interviewer">
          Let's go deep on the model architecture. Walk me through retrieval and
          ranking mathematically.
        </Speaker>

        <Speaker role="candidate">
          <strong>
            Stage 1: Two-Tower Retrieval Model
          </strong>
          <br />
          <br />
          The core idea: learn separate embedding functions for users and events
          such that their dot product approximates relevance.
        </Speaker>

        <Diagram title="Two-Tower Architecture">
          {`
  User Features                              Event Features
  ┌─────────────┐                            ┌──────────────┐
  │ user_id     │                            │ event_id     │
  │ city        │                            │ category     │
  │ past_RSVPs  │                            │ title_emb    │
  │ age_bucket  │                            │ price_bucket │
  │ pref_cats   │                            │ venue_geo    │
  └──────┬──────┘                            └──────┬───────┘
         │                                          │
         ▼                                          ▼
  ┌──────────────┐                           ┌──────────────┐
  │  Embedding   │                           │  Embedding   │
  │  Layers      │                           │  Layers      │
  └──────┬───────┘                           └──────┬───────┘
         │                                          │
         ▼                                          ▼
  ┌──────────────┐                           ┌──────────────┐
  │  MLP Layers  │                           │  MLP Layers  │
  │  512→256→128 │                           │  512→256→128 │
  │  + BatchNorm │                           │  + BatchNorm │
  │  + ReLU      │                           │  + ReLU      │
  └──────┬───────┘                           └──────┬───────┘
         │                                          │
         ▼                                          ▼
     uₑ ∈ ℝ¹²⁸                               eₑ ∈ ℝ¹²⁸
     (user emb)                              (event emb)
         │                                          │
         └──────────────┬───────────────────────────┘
                        │
                        ▼
               sim(u, e) = uᵀe / τ
               (cosine similarity / temperature)`}
        </Diagram>

        <Speaker role="candidate">
          <strong>Training objective:</strong> We use{" "}
          <strong>in-batch sampled softmax</strong> with temperature scaling.
          Given a batch of N (user, event) positive pairs, for each user uᵢ, the
          loss is:
        </Speaker>

        <MathBlock>
          L(uᵢ) = −log( exp(uᵢᵀeᵢ / τ) / Σⱼ₌₁ᴺ exp(uᵢᵀeⱼ / τ) )
        </MathBlock>

        <Speaker role="candidate">
          Where τ (temperature) is typically 0.05–0.1. The denominator sums
          over all events in the batch — the other N−1 events serve as{" "}
          <strong>implicit negatives</strong>.
          <br />
          <br />
          <strong>Key issue: popularity bias correction.</strong> In-batch
          negatives are sampled proportional to their frequency in training data,
          so popular events appear disproportionately as negatives. This causes
          the model to under-recommend popular items (the logQ correction from
          the YouTube paper):
        </Speaker>

        <MathBlock>
          corrected_logit(uᵢ, eⱼ) = uᵢᵀeⱼ / τ − log(pⱼ){"\n"}
          where pⱼ = frequency of event j in training data / total events
        </MathBlock>

        <Speaker role="candidate">
          Without this correction, the model learns to penalize popular events
          because they appear as negatives too often. The YouTube DNN paper
          showed this can drop Recall@K by 5-10%.
          <br />
          <br />
          <strong>
            Stage 3: Ranking Model — Deep Cross Network v2 (DCN-v2) with Multi-Task
          </strong>
          <br />
          <br />
          The ranking model is fundamentally different from retrieval: it sees
          user-event <em>pairs</em> and can compute cross-features. I'd use a
          DCN-v2 architecture with multi-task heads.
        </Speaker>

        <Diagram title="DCN-v2 Multi-Task Ranking Model">
          {`
    ┌─────────────────────────────────────────────────┐
    │           Input Feature Layer                    │
    │  [user_features ⊕ event_features ⊕ cross_feats] │
    │  concat → x₀ ∈ ℝᵈ                               │
    └─────────────────────┬───────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
    ┌──────────────────┐    ┌──────────────────┐
    │   Cross Network  │    │   Deep Network   │
    │                  │    │                  │
    │  xₗ₊₁ = x₀ ⊙   │    │  MLP: 1024→512   │
    │   (Wₗ·xₗ + bₗ)  │    │  →256→128        │
    │   + xₗ           │    │  ReLU + Dropout  │
    │                  │    │  (0.1)           │
    │  (3 cross layers)│    │                  │
    └────────┬─────────┘    └────────┬─────────┘
             │                       │
             └───────────┬───────────┘
                         │ concat
                         ▼
              ┌──────────────────────┐
              │    Shared Layer      │
              │    256 → 128         │
              └──────────┬───────────┘
                         │
           ┌─────────────┼─────────────┐
           │             │             │
           ▼             ▼             ▼
    ┌────────────┐ ┌──────────┐ ┌───────────┐
    │ P(click)   │ │ P(RSVP)  │ │ P(attend) │
    │ σ(wc·h+bc)│ │ σ(wr·h+br│ │ σ(wa·h+ba)│
    │ (sigmoid)  │ │ (sigmoid)│ │ (sigmoid) │
    └────────────┘ └──────────┘ └───────────┘`}
        </Diagram>

        <Speaker role="candidate">
          <strong>Why DCN-v2?</strong>
          <br />
          <br />
          The cross network explicitly models feature interactions like
          "user_preferred_price × event_price × day_of_week" without manual
          feature crossing. Each cross layer computes:
        </Speaker>

        <MathBlock>
          xₗ₊₁ = x₀ ⊙ (Wₗ · xₗ + bₗ) + xₗ
        </MathBlock>

        <Speaker role="candidate">
          Where ⊙ is element-wise multiplication. This is bounded-degree
          polynomial feature interaction — layer l captures up to (l+1)-order
          interactions. With 3 cross layers, we get up to 4th-order interactions
          at linear cost. Compare this to a pure DNN which approximates these
          interactions less efficiently.
          <br />
          <br />
          <strong>Multi-Task Training Loss:</strong>
        </Speaker>

        <MathBlock>
          L = λ₁·BCE(ŷ_click, y_click) + λ₂·BCE(ŷ_rsvp, y_rsvp) +
          λ₃·BCE(ŷ_attend, y_attend){"\n"}
          {"\n"}
          where BCE(ŷ, y) = −[y·log(ŷ) + (1−y)·log(1−ŷ)]{"\n"}
          {"\n"}
          Typical weights: λ₁ = 0.2, λ₂ = 0.5, λ₃ = 0.3
        </MathBlock>

        <Speaker role="candidate">
          The multi-task setup has two big advantages:
          <br />
          <br />
          <strong>1. Shared representation:</strong> The click task has
          abundant data (millions/day) and helps learn good lower-layer features
          that transfer to the RSVP task (sparser, thousands/day) and attend
          task (sparsest, hundreds/day). This is essentially a form of{" "}
          <strong>auxiliary task regularization</strong>.
          <br />
          <br />
          <strong>2. Calibrated multi-objective scoring:</strong> At serving
          time, the final score combines all three predictions.
        </Speaker>

        <MathBlock>
          final_score = w₁·P(click) + w₂·P(RSVP|click)·P(click) +
          w₃·P(attend|RSVP)·P(RSVP|click)·P(click){"\n"}
          {"\n"}≈ w₁·P(click) + w₂·P(RSVP) + w₃·P(attend)
        </MathBlock>

        <Speaker role="interviewer">
          Good. What about the training data? How do you construct positive and
          negative labels?
        </Speaker>

        <Speaker role="candidate">
          <strong>For retrieval (two-tower):</strong>
          <br />• <em>Positives:</em> (user, event) pairs where user RSVP'd or
          purchased a ticket
          <br />• <em>Negatives:</em> In-batch negatives (all other events in
          the batch) + hard negatives sampled from events the user saw but
          didn't click
          <br />
          <br />
          <strong>For ranking (DCN-v2):</strong>
          <br />• Training data comes from <em>logged impressions</em> — events
          that were actually shown to users
          <br />• Label is click=1/0, RSVP=1/0, attend=1/0
          <br />• Critical: only train on events the user actually <em>saw</em>,
          not all events. This avoids <strong>selection bias</strong>.
          <br />
          <br />
          <strong>Hard negative mining</strong> is crucial for retrieval quality.
          I'd use a mix of:
          <br />
          <br />
          <strong>1.</strong> In-batch negatives (easy, free)
          <br />
          <strong>2.</strong> Events retrieved by the model but not clicked
          (semi-hard)
          <br />
          <strong>3.</strong> Events in same category/geo but not interacted with
          (hard)
        </Speaker>

        <Callout type="tradeoff" title="Tradeoff — DCN-v2 vs. DeepFM vs. Transformer">
          <strong>DCN-v2:</strong> Explicit bounded-degree feature crosses.
          Efficient, interpretable crosses. Best when feature interactions are
          important but you want controlled complexity. This is my pick.
          <br />
          <strong>DeepFM:</strong> Factorization machine + DNN. Good for sparse
          features. Slightly weaker on high-order interactions.
          <br />
          <strong>Transformer-based (BST, DIN):</strong> Great for modeling
          user behavior sequences ("attended jazz → clicked blues → ?"). Higher
          latency (~2-5x DCN). Worth it if sequential behavior is a dominant
          signal. I'd consider adding a DIN-style attention layer on top of
          DCN-v2 for the user's recent interaction sequence.
        </Callout>

        <Speaker role="interviewer">
          How do you handle the class imbalance? RSVP rate might be ~2% and
          attend rate ~0.5%.
        </Speaker>

        <Speaker role="candidate">
          Three complementary approaches:
          <br />
          <br />
          <strong>1. Negative downsampling:</strong> Randomly sample negatives at
          rate α (say 0.1), then correct the prediction at serving time:
        </Speaker>

        <MathBlock>
          P_corrected = P_model / (P_model + (1 - P_model) / α)
        </MathBlock>

        <Speaker role="candidate">
          This reduces training data size by ~10x without losing signal,
          which is what Google's Wide & Deep paper recommends.
          <br />
          <br />
          <strong>2. Focal loss</strong> for the attend task (very sparse):
        </Speaker>

        <MathBlock>
          FL(pₜ) = −αₜ (1 − pₜ)ᵞ · log(pₜ)    where γ = 2
        </MathBlock>

        <Speaker role="candidate">
          The (1−pₜ)ᵞ term down-weights easy negatives that the model is already
          confident about, focusing learning on the hard cases.
          <br />
          <br />
          <strong>3. Task weighting</strong> via uncertainty-based multi-task
          loss (Kendall et al.):
        </Speaker>

        <MathBlock>
          L = Σₜ (1/2σₜ²) · Lₜ + log(σₜ){"\n"}
          where σₜ is a learned per-task uncertainty parameter
        </MathBlock>

        <Speaker role="candidate">
          This automatically balances the loss across tasks — the attend task
          (high uncertainty) gets a smaller effective weight initially, preventing
          it from destabilizing training.
        </Speaker>
      </Section>

      {/* ═══════════ PHASE 6: SERVING ═══════════ */}
      <Section title="Serving & Infrastructure" time="5–8 min" phase="06">
        <Speaker role="interviewer">
          How do you serve this in production at 1000 QPS?
        </Speaker>

        <Speaker role="candidate">
          Let me walk through the serving stack:
        </Speaker>

        <Diagram title="Online Serving Architecture">
          {`
User Request (user_id, lat/lng, timestamp)
         │
         ▼
┌──────────────────────────────────────┐
│         API Gateway / LB             │
│         (rate limiting, auth)        │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│      Recommendation Orchestrator     │
│  (async parallel execution)          │
│                                      │
│  ┌─────────┐ ┌──────────┐ ┌───────┐ │
│  │ Feature  │ │ Geo      │ │ User  │ │
│  │ Store    │ │ Filter   │ │ Emb.  │ │
│  │ Lookup   │ │ Service  │ │ Compute│ │
│  │ (Redis)  │ │ (PostGIS)│ │       │ │
│  └────┬─────┘ └────┬─────┘ └───┬───┘ │
│       └─────────────┼───────────┘     │
│                     ▼                 │
│  ┌─────────────────────────────────┐  │
│  │   ANN Index (ScaNN / Milvus)   │  │
│  │   Returns top-500 event IDs    │  │
│  └──────────────┬──────────────────┘  │
│                 ▼                     │
│  ┌─────────────────────────────────┐  │
│  │   Pre-Ranker (TF Serving lite) │  │
│  │   Scores 500 → keeps 100       │  │
│  └──────────────┬──────────────────┘  │
│                 ▼                     │
│  ┌─────────────────────────────────┐  │
│  │   Ranker (TF Serving / GPU)    │  │
│  │   Scores 100 items, 3 heads    │  │
│  └──────────────┬──────────────────┘  │
│                 ▼                     │
│  ┌─────────────────────────────────┐  │
│  │   Re-Ranker (rule engine)      │  │
│  │   Diversity, dedup, freshness  │  │
│  └──────────────┬──────────────────┘  │
│                 │                     │
└─────────────────┼─────────────────────┘
                  ▼
         JSON: [event_1, event_2, ...]`}
        </Diagram>

        <Speaker role="candidate">
          Key infrastructure decisions:
          <br />
          <br />
          <strong>Feature Store (Feast + Redis):</strong> User features and event
          features are precomputed offline and stored in Redis for &lt;1ms
          lookups. Cross-features (like user-event distance) are computed online
          during the ranking stage.
          <br />
          <br />
          <strong>ANN Index:</strong> I'd use <strong>ScaNN</strong> (Google) or{" "}
          <strong>HNSW</strong> (via Milvus/Vespa). For 20K geo-filtered events
          with 128-dim embeddings, query time is ~2ms. The index is rebuilt
          incrementally every 30-60 minutes as new events are created.
          <br />
          <br />
          <strong>Model Serving:</strong> Ranking model on TensorFlow Serving
          with batching enabled (batch size 32, max wait 5ms). This amortizes
          GPU compute. Pre-ranker on CPU (it's lightweight enough).
          <br />
          <br />
          <strong>Caching:</strong> Two-layer cache:
          <br />
          • L1: User-level result cache (TTL=5min) — if the same user refreshes
          within 5 min, serve cached results
          <br />• L2: User embedding cache (TTL=1hr) — avoid recomputing the
          user tower for every request
        </Speaker>

        <Speaker role="interviewer">
          What about the ANN index — how do you handle the fact that events
          expire?
        </Speaker>

        <Speaker role="candidate">
          This is the trickiest infra challenge specific to events. Two approaches:
          <br />
          <br />
          <strong>Option A: Filtered ANN search.</strong> Store all events in the
          index with metadata (expiry_time, remaining_seats, geo_hash). At query
          time, apply a pre-filter on metadata before ANN search. Milvus and
          Vespa support this natively. Downside: filtering reduces effective
          index size and can hurt recall.
          <br />
          <br />
          <strong>Option B: Time-partitioned indexes.</strong> Maintain separate
          ANN indexes per time window (events this week, next week, next month).
          Query all relevant indexes in parallel and merge results. This avoids
          filtering overhead but adds operational complexity.
          <br />
          <br />
          I'd go with <strong>Option A</strong> for simplicity. With 20K
          geo-filtered candidates (after the geo pre-filter step), even with
          metadata filtering, the ANN search completes in &lt;5ms.
        </Speaker>

        <Callout type="mental" title="Mental Model — ANN Index Refresh Strategy">
          Think of ANN indexes on a spectrum:
          <br />
          <strong>Batch (FAISS flat rebuild):</strong> Rebuild entirely from
          scratch periodically. Simple but stale. OK for product catalogs.
          <br />
          <strong>Incremental (HNSW, Milvus):</strong> Insert/delete individual
          vectors. Near-real-time freshness. More complex to maintain but
          essential for fast-changing inventories like events.
          <br />
          <strong>Streaming (Vespa, custom):</strong> Events indexed within
          seconds of creation. Maximum freshness at maximum complexity.
          <br />
          <br />
          For events, you need at least "Incremental." The refresh cadence
          should match your business SLA — if an organizer expects their event
          to appear in recommendations within 1 hour, your pipeline must index
          it within 1 hour.
        </Callout>
      </Section>

      {/* ═══════════ PHASE 7: MONITORING ═══════════ */}
      <Section
        title="Monitoring, Iteration & Edge Cases"
        time="5 min"
        phase="07"
      >
        <Speaker role="interviewer">
          How do you monitor this system and iterate on it?
        </Speaker>

        <Speaker role="candidate">
          <strong>Online Monitoring:</strong>
          <br />
          <br />
          • <em>Real-time dashboards:</em> CTR, RSVP rate, p50/p99 latency per
          stage, ANN index freshness lag
          <br />• <em>Alerting on:</em> Latency spikes (&gt;300ms p99), CTR
          drops &gt;10% from baseline, model serving errors, feature store stale
          data
          <br />• <em>Data quality monitors:</em> Feature distribution drift
          (KL-divergence between training and serving distributions), null rate
          monitoring
          <br />
          <br />
          <strong>Model Monitoring:</strong>
          <br />
          <br />
          • <em>Prediction calibration:</em> Is P(RSVP)=0.05 actually resulting
          in 5% RSVP rate? Plot calibration curves daily.
          <br />• <em>Feature importance drift:</em> If the model suddenly relies
          heavily on a single feature, investigate data issues.
          <br />• <em>A/B testing framework:</em> I'd use a holdout-based system
          where 5% of traffic sees the existing model and 5% sees the new model,
          testing on RSVP rate as the primary metric with 95% confidence.
          <br />
          <br />
          <strong>Retraining Cadence:</strong>
          <br />
          • Retrieval model: retrain weekly (embeddings are relatively stable)
          <br />• Ranking model: retrain daily with the last 30 days of data (it
          needs fresh behavioral signals)
          <br />• Use <strong>warm-starting</strong>: initialize from last
          checkpoint and fine-tune on new data, rather than training from scratch
        </Speaker>

        <Speaker role="interviewer">
          What are some failure modes or edge cases?
        </Speaker>

        <Speaker role="candidate">
          <strong>1. Filter bubble:</strong> User only attends tech meetups →
          system only recommends tech meetups → user never discovers cooking
          classes they'd love. Solution: inject an <strong>exploration
          component</strong> in re-ranking — 10-15% of slots filled with
          "contextually adjacent" categories (e.g., if tech → design meetups,
          entrepreneurship events).
          <br />
          <br />
          <strong>2. Popularity bias amplification:</strong> Popular events get
          more clicks → more training signal → ranked higher → more clicks.
          Solution: add a <strong>popularity penalty</strong> in re-ranking:
          score_final = score_model × (1 / log(1 + total_RSVPs)^β), where β
          controls the penalty strength.
          <br />
          <br />
          <strong>3. Temporal exploitation:</strong> Events happening tomorrow
          get urgency clicks, not genuine interest. Solution: include
          "days_until_event" as a feature and be careful not to overweight
          imminent events.
          <br />
          <br />
          <strong>4. Organizer fairness:</strong> New organizers with no history
          get buried. Solution: organizer-level exposure guarantees — ensure
          every organizer gets a minimum number of impressions proportional to
          their event count (a form of multi-sided fairness).
          <br />
          <br />
          <strong>5. Social signal leakage:</strong> "3 friends attending" is a
          powerful feature but creates a rich-get-richer dynamic for socially
          connected events. We should A/B test the marginal value of the social
          signal vs. its concentration effects.
        </Speaker>

        <Callout type="mental" title="Mental Model — The Exploration-Exploitation Dial">
          Every recommender system sits on a spectrum between{" "}
          <em>exploitation</em> (showing what you know the user likes) and{" "}
          <em>exploration</em> (showing novel items to learn more). For events,
          exploration is <em>more important</em> than for products because: (a)
          events are ephemeral — if you don't explore now, the event expires,
          and (b) user preferences for events are less stable than for products
          (someone might want jazz one week and hiking the next). A good rule of
          thumb: 15-20% exploration for events vs. 5-10% for products.
        </Callout>
      </Section>

      {/* ═══════════ SUMMARY ═══════════ */}
      <Section title="Summary & Key Takeaways" time="Wrap-up" phase="★">
        <Diagram title="Complete System — One-Slide Summary">
          {`
  ┌─────────────────── OFFLINE ──────────────────┐
  │                                               │
  │  Click/RSVP/Attend Logs                       │
  │       │                                       │
  │       ▼                                       │
  │  Feature Engineering ──→ Feature Store (Feast) │
  │       │                                       │
  │       ▼                                       │
  │  Two-Tower Training ──→ Event ANN Index        │
  │  (weekly, sampled softmax + logQ correction)  │
  │       │                                       │
  │       ▼                                       │
  │  DCN-v2 Multi-Task Training (daily)           │
  │  (P(click), P(RSVP), P(attend))               │
  │  focal loss + uncertainty-weighted MTL         │
  │       │                                       │
  │       ▼                                       │
  │  Model Registry → Canary Deploy → Full Rollout│
  │                                               │
  └───────────────────────────────────────────────┘
                        │
                        ▼
  ┌─────────────────── ONLINE ───────────────────┐
  │                                               │
  │  User Request → Geo Filter (500K→20K)         │
  │       │                                       │
  │       ▼                                       │
  │  Two-Tower Retrieval + ANN (20K→500) [<20ms]  │
  │       │                                       │
  │       ▼                                       │
  │  Pre-Ranker: distilled model (500→100) [<15ms]│
  │       │                                       │
  │       ▼                                       │
  │  DCN-v2 Ranker: 3-head scoring (100→50) [<50ms│
  │       │                                       │
  │       ▼                                       │
  │  Re-Rank: diversity, freshness, fairness      │
  │       │                                       │
  │       ▼                                       │
  │  Response: Top 50 events [< 200ms p99 total]  │
  │                                               │
  └───────────────────────────────────────────────┘`}
        </Diagram>

        <div className="mt-4 space-y-3 text-sm">
          <div className="p-3 bg-indigo-50 rounded-lg border border-indigo-200">
            <strong className="text-indigo-800">Key Design Decisions:</strong>
            <br />
            <span className="text-indigo-700">
              Two-tower for retrieval (cold-start friendly) → DCN-v2 for ranking
              (explicit feature crosses) → Multi-task heads (P(click), P(RSVP),
              P(attend)) → Re-ranking for business constraints
            </span>
          </div>
          <div className="p-3 bg-emerald-50 rounded-lg border border-emerald-200">
            <strong className="text-emerald-800">
              What Makes Events Unique:
            </strong>
            <br />
            <span className="text-emerald-700">
              Temporal perishability requires incremental ANN indexes • Geo
              constraints enable a cheap pre-filter stage • Severe cold-start
              demands content-based embeddings over collaborative signals • High
              exploration rate (15-20%) because preferences are less stable
            </span>
          </div>
          <div className="p-3 bg-amber-50 rounded-lg border border-amber-200">
            <strong className="text-amber-800">Interviewer Scoring Rubric:</strong>
            <br />
            <span className="text-amber-700">
              ✓ Clarified domain-specific constraints (not generic RecSys){" "}
              ✓ Multi-level metrics with composite objective{" "}
              ✓ Multi-stage funnel with latency budget{" "}
              ✓ Mathematical depth (loss functions, corrections){" "}
              ✓ Cold-start strategy leveraging two-tower architecture{" "}
              ✓ Production considerations (ANN refresh, caching, monitoring){" "}
              ✓ Edge cases and fairness awareness
            </span>
          </div>
        </div>
      </Section>
    </div>
  );
}
