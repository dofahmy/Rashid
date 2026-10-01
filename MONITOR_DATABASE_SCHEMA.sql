-- Reference schema only. Application creates additive tables automatically.

CREATE TABLE activities (
	id SERIAL NOT NULL, 
	lead_id INTEGER NOT NULL, 
	kind VARCHAR(30) NOT NULL, 
	body TEXT, 
	created_at VARCHAR(40), 
	PRIMARY KEY (id)
)

;
CREATE INDEX ix_activities_lead_id ON activities (lead_id);

CREATE TABLE leads (
	id SERIAL NOT NULL, 
	telegram_id BIGINT NOT NULL, 
	username VARCHAR(100), 
	telegram_name VARCHAR(200), 
	name VARCHAR(150), 
	phone VARCHAR(25), 
	market VARCHAR(10), 
	draft_name VARCHAR(150), 
	draft_phone VARCHAR(25), 
	draft_market VARCHAR(10), 
	step VARCHAR(20), 
	source VARCHAR(100), 
	status VARCHAR(20), 
	owner VARCHAR(100), 
	follow_up VARCHAR(20), 
	notes TEXT, 
	consent_at VARCHAR(40), 
	consent_text TEXT, 
	completed_at VARCHAR(40), 
	created_at VARCHAR(40), 
	updated_at VARCHAR(40), 
	last_message_at VARCHAR(40), 
	telegram_status VARCHAR(20), 
	PRIMARY KEY (id), 
	UNIQUE (telegram_id)
)

;
CREATE INDEX ix_leads_completed_at ON leads (completed_at);
CREATE INDEX ix_leads_created_at ON leads (created_at);
CREATE INDEX ix_leads_source ON leads (source);
CREATE INDEX ix_leads_status ON leads (status);
CREATE INDEX ix_leads_phone ON leads (phone);

CREATE TABLE market_candles_15m (
	symbol VARCHAR(40) NOT NULL, 
	ts BIGINT NOT NULL, 
	o FLOAT NOT NULL, 
	h FLOAT NOT NULL, 
	l FLOAT NOT NULL, 
	c FLOAT NOT NULL, 
	v FLOAT NOT NULL, 
	PRIMARY KEY (symbol, ts)
)

;

CREATE TABLE market_events (
	id SERIAL NOT NULL, 
	key VARCHAR(150) NOT NULL, 
	plan_id INTEGER NOT NULL, 
	symbol VARCHAR(40) NOT NULL, 
	kind VARCHAR(40) NOT NULL, 
	bar_ts BIGINT NOT NULL, 
	details_json TEXT, 
	created_at VARCHAR(40), 
	PRIMARY KEY (id), 
	UNIQUE (key)
)

;
CREATE INDEX ix_market_events_plan_id ON market_events (plan_id);

CREATE TABLE market_plans (
	id SERIAL NOT NULL, 
	symbol VARCHAR(40) NOT NULL, 
	market VARCHAR(2) NOT NULL, 
	strategy_version VARCHAR(80) NOT NULL, 
	state VARCHAR(30) NOT NULL, 
	signal_ts BIGINT NOT NULL, 
	last_bar BIGINT NOT NULL, 
	entry FLOAT NOT NULL, 
	stop FLOAT NOT NULL, 
	target FLOAT NOT NULL, 
	atr FLOAT NOT NULL, 
	score FLOAT NOT NULL, 
	waiting_bars INTEGER, 
	retest_bars INTEGER, 
	trigger_ts BIGINT, 
	activation_ts BIGINT, 
	paper_entry FLOAT, 
	exit_price FLOAT, 
	context_json TEXT NOT NULL, 
	policy_json TEXT NOT NULL, 
	created_at VARCHAR(40), 
	updated_at VARCHAR(40), 
	PRIMARY KEY (id), 
	UNIQUE (symbol, strategy_version, signal_ts)
)

;
CREATE INDEX ix_market_plans_market ON market_plans (market);
CREATE INDEX ix_market_plans_symbol ON market_plans (symbol);
CREATE INDEX ix_market_plans_state ON market_plans (state);

CREATE TABLE market_scans (
	id SERIAL NOT NULL, 
	market VARCHAR(2) NOT NULL, 
	boundary BIGINT NOT NULL, 
	status VARCHAR(30), 
	expected_bar BIGINT, 
	total INTEGER, 
	ok INTEGER, 
	errors INTEGER, 
	new_plans INTEGER, 
	transitions INTEGER, 
	started_at VARCHAR(40), 
	finished_at VARCHAR(40), 
	summary_json TEXT, 
	PRIMARY KEY (id)
)

;

CREATE TABLE market_stocks (
	symbol VARCHAR(40) NOT NULL, 
	feed_symbol VARCHAR(40) NOT NULL, 
	market VARCHAR(2) NOT NULL, 
	company VARCHAR(250) NOT NULL, 
	metadata_json TEXT NOT NULL, 
	sharia_label VARCHAR(150), 
	last_bar BIGINT, 
	last_price FLOAT, 
	checked_at VARCHAR(40), 
	error VARCHAR(120), 
	evaluation_json TEXT, 
	PRIMARY KEY (symbol)
)

;
CREATE INDEX ix_market_stocks_market ON market_stocks (market);

CREATE TABLE outbox (
	id SERIAL NOT NULL, 
	key VARCHAR(150) NOT NULL, 
	chat_id BIGINT NOT NULL, 
	method VARCHAR(40), 
	payload TEXT NOT NULL, 
	status VARCHAR(20), 
	attempts INTEGER, 
	next_at BIGINT, 
	error VARCHAR(120), 
	PRIMARY KEY (id), 
	UNIQUE (key)
)

;
CREATE INDEX ix_outbox_status ON outbox (status);

CREATE TABLE processed_updates (
	id BIGSERIAL NOT NULL, 
	PRIMARY KEY (id)
)

;

CREATE TABLE settings (
	key VARCHAR(80) NOT NULL, 
	value TEXT, 
	PRIMARY KEY (key)
)

;

CREATE TABLE us_recommendation_preferences (
	telegram_id BIGSERIAL NOT NULL, 
	paused INTEGER NOT NULL, 
	PRIMARY KEY (telegram_id)
)

;

CREATE TABLE us_recommendation_publications (
	plan_id SERIAL NOT NULL, 
	published_ts BIGINT NOT NULL, 
	activation_end BIGINT NOT NULL, 
	PRIMARY KEY (plan_id)
)

;
CREATE INDEX ix_us_recommendation_publications_published_ts ON us_recommendation_publications (published_ts);

CREATE TABLE us_recommendation_recipients (
	plan_id INTEGER NOT NULL, 
	telegram_id BIGINT NOT NULL, 
	entry_key VARCHAR(150) NOT NULL, 
	PRIMARY KEY (plan_id, telegram_id), 
	UNIQUE (entry_key)
)

;