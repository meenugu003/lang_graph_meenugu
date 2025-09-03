/* =========================
   Product Decision Trees
   ========================= */

/* Metadata for each decision tree */
CREATE TABLE decision_trees (
    product TEXT,
    version TEXT,
    tree_id UUID,
    bundle_id UUID,
    source_bundle TEXT,       -- optional string reference (e.g., "bndl-123")
    created_at TIMESTAMP,
    PRIMARY KEY ((product), version, tree_id)
) WITH CLUSTERING ORDER BY (version DESC);


/* Nodes for each decision tree (questions & leaves) */
CREATE TABLE decision_tree_nodes (
    tree_id UUID,
    node_id TEXT,
    kind TEXT,                -- "question" or "leaf"
    text TEXT,                -- question text (nullable for leaf)
    emits_pattern TEXT,       -- strategy pattern (nullable for question)
    options LIST<FROZEN<MAP<TEXT, TEXT>>>, -- [{label: "Yes", next: "n2"}]
    PRIMARY KEY ((tree_id), node_id)
);


/* Provenance: which docs (files) support each node */
CREATE TABLE provenance (
    tree_id UUID,
    node_id TEXT,
    file_id UUID,
    score DOUBLE,
    PRIMARY KEY ((tree_id), node_id, file_id)
);


/* =========================
   File Uploads
   ========================= */

/* Upload bundle = one upload event (multiple files) */
CREATE TABLE upload_bundles (
    bundle_id UUID,
    product TEXT,
    uploaded_by TEXT,
    uploaded_at TIMESTAMP,
    version TEXT,
    tree_id UUID,             -- decision tree generated from this bundle
    PRIMARY KEY ((product), bundle_id)
) WITH CLUSTERING ORDER BY (bundle_id DESC);


/* Files in each bundle */
CREATE TABLE upload_files (
    bundle_id UUID,
    file_id UUID,
    file_name TEXT,
    file_path TEXT,           -- storage reference (S3/GCS/local path)
    content TEXT,             -- optional: only if storing file content in Cassandra
    PRIMARY KEY ((bundle_id), file_id)
);


/* =========================
   User Sessions
   ========================= */

/* Tracks user conversation progress with chatbot */
CREATE TABLE user_sessions (
    user_id TEXT,
    session_id UUID,
    product TEXT,
    tree_id UUID,
    current_node_id TEXT,
    history LIST<FROZEN<MAP<TEXT, TEXT>>>, -- [{node_id: "n1", choice: "Yes"}]
    updated_at TIMESTAMP,
    PRIMARY KEY ((user_id), session_id)
) WITH CLUSTERING ORDER BY (session_id DESC);
