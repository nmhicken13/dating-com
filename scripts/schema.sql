-- Run once in Supabase Dashboard → SQL → New query → Run
-- (Same as scripts/create_supabase_tables.py)
-- If you previously created `date_photos`, you can remove it with:
--   DROP TABLE IF EXISTS public.date_photos CASCADE;

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS public.people (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users (id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    initial_met_via TEXT,
    profile_image TEXT,
    roster_slot INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_people_user_id ON public.people (user_id);
CREATE INDEX IF NOT EXISTS idx_people_user_name ON public.people (user_id, name);

CREATE TABLE IF NOT EXISTS public.dates (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users (id) ON DELETE CASCADE,
    person_id BIGINT NOT NULL REFERENCES public.people (id) ON DELETE CASCADE,
    occurred_on DATE NOT NULL,
    activity TEXT,
    notes TEXT,
    rating INTEGER CHECK (rating IS NULL OR (rating >= 1 AND rating <= 10)),
    physical_escalation TEXT,
    outing_type TEXT NOT NULL DEFAULT 'date'
        CHECK (outing_type IN ('date', 'casual')),
    company_type TEXT NOT NULL DEFAULT 'one_on_one'
        CHECK (company_type IN ('one_on_one', 'double', 'group')),
    thank_you INTEGER CHECK (thank_you IS NULL OR thank_you IN (0, 1)),
    cost DOUBLE PRECISION CHECK (cost IS NULL OR cost >= 0),
    is_planned INTEGER NOT NULL DEFAULT 0 CHECK (is_planned IN (0, 1)),
    scheduled_date TEXT,
    initiator TEXT,
    duration_hours DOUBLE PRECISION CHECK (duration_hours IS NULL OR duration_hours >= 0),
    user_wanted_next_date INTEGER NOT NULL DEFAULT 1 CHECK (user_wanted_next_date IN (0, 1)),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_dates_user_id ON public.dates (user_id);
CREATE INDEX IF NOT EXISTS idx_dates_person ON public.dates (person_id);
CREATE INDEX IF NOT EXISTS idx_dates_occurred ON public.dates (occurred_on);

CREATE TABLE IF NOT EXISTS public.ml_configs (
    user_id UUID PRIMARY KEY REFERENCES auth.users (id) ON DELETE CASCADE,
    coef JSONB NOT NULL,
    mean JSONB NOT NULL,
    scale JSONB NOT NULL,
    intercept DOUBLE PRECISION NOT NULL,
    feature_names JSONB,
    metrics JSONB,
    heuristic_metrics JSONB,
    calibration JSONB,
    heuristic_calibration JSONB,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE public.people ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.dates ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ml_configs ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS people_own ON public.people;
CREATE POLICY people_own ON public.people
    FOR ALL TO authenticated
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS dates_own ON public.dates;
CREATE POLICY dates_own ON public.dates
    FOR ALL TO authenticated
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS ml_configs_own ON public.ml_configs;
CREATE POLICY ml_configs_own ON public.ml_configs
    FOR ALL TO authenticated
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

GRANT USAGE ON SCHEMA public TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO authenticated;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO authenticated;
