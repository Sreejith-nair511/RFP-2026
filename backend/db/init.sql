-- DeceptiScope v2 — PostgreSQL schema
-- Tracks experiments, evaluation runs, and deception analysis results

CREATE TABLE IF NOT EXISTS experiments (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT,
    model_name      TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    config          JSONB
);

CREATE TABLE IF NOT EXISTS conversations (
    id              SERIAL PRIMARY KEY,
    experiment_id   INTEGER REFERENCES experiments(id),
    model_name      TEXT NOT NULL,
    started_at      TIMESTAMPTZ DEFAULT NOW(),
    ended_at        TIMESTAMPTZ,
    metadata        JSONB
);

CREATE TABLE IF NOT EXISTS messages (
    id              SERIAL PRIMARY KEY,
    conversation_id INTEGER REFERENCES conversations(id),
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    timestamp       TIMESTAMPTZ DEFAULT NOW(),
    deception_score FLOAT,
    deception_type  TEXT,
    confidence      FLOAT,
    type_scores     JSONB,
    signal_contributions JSONB,
    per_token_scores JSONB,
    high_risk_tokens JSONB
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id              SERIAL PRIMARY KEY,
    experiment_id   INTEGER REFERENCES experiments(id),
    benchmark_name  TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    started_at      TIMESTAMPTZ DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    auc_roc         FLOAT,
    accuracy        FLOAT,
    precision       FLOAT,
    recall          FLOAT,
    f1_score        FLOAT,
    calibration_error FLOAT,
    steering_improvement FLOAT,
    baseline_comparisons JSONB,
    type_performance JSONB,
    full_results    JSONB
);

CREATE TABLE IF NOT EXISTS shadow_model_checkpoints (
    id              SERIAL PRIMARY KEY,
    target_model    TEXT NOT NULL,
    base_model      TEXT NOT NULL,
    checkpoint_path TEXT NOT NULL,
    fidelity_score  FLOAT,
    training_pairs  INTEGER,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    metadata        JSONB
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_deception_score ON messages(deception_score);
CREATE INDEX IF NOT EXISTS idx_eval_runs_benchmark ON eval_runs(benchmark_name, model_name);
CREATE INDEX IF NOT EXISTS idx_shadow_checkpoints_target ON shadow_model_checkpoints(target_model);
