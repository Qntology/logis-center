CREATE TABLE IF NOT EXISTS usage (
    "id" TEXT PRIMARY KEY,
    "type" TEXT,
    "user" TEXT,
    "team" TEXT,
    "zone" REAL,
    'date' TEXT
);

CREATE TABLE IF NOT EXISTS items (
    "id" TEXT PRIMARY KEY,
    "type" TEXT,
    "from" TEXT,
    "to" TEXT,
    "cc" TEXT,
    "bcc" TEXT,
    "ref" TEXT,
    "digest" TEXT,
    "data" BLOB NULL,
    "created_at" INTEGER,
    "updated_at" INTEGER
);


CREATE TABLE IF NOT EXISTS sales (
    "id" TEXT PRIMARY KEY,
    "type" TEXT,
    "from" TEXT,
    "to" TEXT,
    "cc" TEXT,
    "bcc" TEXT,
    "ref" TEXT,
    "data" BLOB NULL,
    "created_at" INTEGER,
    "started_at" INTEGER,
    "expired_at" INTEGER,
    "index" INTEGER,
    "event" INTEGER,
    "views" INTEGER,
    "goods" INTEGER,
    "status" INTEGER,
    "width" REAL,
    "height" REAL,
    "length" REAL,
    "weight" REAL,
    "size" TEXT,
    "currency" TEXT,
    "supply_price" REAL,
    "sale_price" REAL,
    "discount" REAL,
    "quantity" INTEGER,
    "tracking" INTEGER,
    "number" TEXT,
    "carrier" TEXT,
    "shipping_fee" REAL,
    "shipping_method" TEXT,
    "shipping_duration" INTEGER,
    "fulfillment_service" TEXT,
    "stock_keeping_unit" TEXT,
    "bundle_shipping" INTEGER,
    "used" INTEGER,
    "lease" INTEGER,
    "rental" INTEGER,
    "refurbish" INTEGER,
    "tax_included" REAL,
    "release_date" INTEGER
);

CREATE TABLE IF NOT EXISTS event (
    "id" TEXT PRIMARY KEY,
    "type" TEXT,
    "from" TEXT,
    "to" TEXT,
    "cc" TEXT,
    "bcc" TEXT,
    "ref" TEXT,
    "data" BLOB NULL,
    "created_at" INTEGER,
    "started_at" INTEGER,
    "expired_at" INTEGER,
    "index" INTEGER,
    "event" INTEGER,
    "phone" TEXT,
    "address" TEXT,
    "status" INTEGER,
    "code" TEXT,
    "discount" REAL,
    "quantity" INTEGER,
    "usage_per" INTEGER,
    "usage_limit" INTEGER,
    "min_order_amount" REAL,
    "max_order_amount" REAL,
    "max_discount_amount" REAL,
    "new_customer_only" INTEGER,
    "first_purchase_only" INTEGER,
    "region_restrictions" INTEGER
);

CREATE TABLE IF NOT EXISTS tracking (
    "id" TEXT PRIMARY KEY,
    "type" TEXT,
    "from" TEXT,
    "to" TEXT,
    "cc" TEXT,
    "bcc" TEXT,
    "ref" TEXT,
    "data" BLOB NULL,
    "created_at" INTEGER,
    "index" INTEGER,
    "event" INTEGER,
    "goods" INTEGER,
    "order" INTEGER,
    "status" INTEGER,
    "no" TEXT,
    "sender_address" TEXT,
    "sender_phone" TEXT,
    "recipient_address" TEXT,
    "recipient_phone" TEXT,
    "width" REAL,
    "height" REAL,
    "length" REAL,
    "weight" REAL,
    "carrier" TEXT,
    "shipping_fee" REAL,
    "shipping_method" TEXT,
    "shipping_duration" REAL,
    "shipping_date" INTEGER,
    "delivery_date" INTEGER,
    "order_date" INTEGER,
    "payment_date" INTEGER,
    "payment_method" TEXT,
    "payment_origin" TEXT,
    "payment_number" TEXT,
    "bundle_shipping" INTEGER
);

CREATE TABLE IF NOT EXISTS talks (
    "id" TEXT PRIMARY KEY,
    "type" TEXT,
    "from" TEXT,
    "to" TEXT,
    "cc" TEXT,
    "bcc" TEXT,
    "ref" TEXT,
    "data" BLOB NULL,
    "created_at" INTEGER,
    "updated_at" INTEGER
);


CREATE TABLE IF NOT EXISTS users (
    "id" TEXT PRIMARY KEY,
    "type" TEXT,
    "from" TEXT,
    "to" TEXT,
    "cc" TEXT,
    "bcc" TEXT,
    "ref" TEXT,
    "data" BLOB NULL,
    "created_at" INTEGER,
    "updated_at" INTEGER
);

CREATE TABLE IF NOT EXISTS pages (
    "id" TEXT PRIMARY KEY,
    "type" TEXT,
    "from" TEXT,
    "to" TEXT,
    "cc" TEXT,
    "bcc" TEXT,
    "ref" TEXT,
    "data" BLOB NULL,
    "created_at" INTEGER,
    "updated_at" INTEGER
);

CREATE TABLE IF NOT EXISTS views (
    "id" TEXT PRIMARY KEY,
    "type" TEXT,
    "flag" TEXT,
    "lang" TEXT,
    "from" TEXT,
    "to" TEXT,
    "cc" TEXT,
    "bcc" TEXT,
    "ref" TEXT,
    "data" BLOB NULL,
    "created_at" INTEGER,
    "updated_at" INTEGER,
    "goods" INTEGER,
    "order" INTEGER,
    "event" INTEGER
);

CREATE TABLE IF NOT EXISTS zones (
    "id" TEXT PRIMARY KEY,
    "pool" INTEGER
);

CREATE TABLE IF NOT EXISTS tasks (
    "id" TEXT PRIMARY KEY,
    "type" TEXT,
    "from" TEXT,
    "to" TEXT,
    "cc" TEXT,
    "bcc" TEXT,
    "ref" TEXT,
    "data" BLOB NULL,
    "created_at" INTEGER,
    "updated_at" INTEGER
);



CREATE TABLE IF NOT EXISTS pages (
    "id" TEXT PRIMARY KEY,
    "type" TEXT,
    "from" TEXT,
    "to" TEXT,
    "cc" TEXT,
    "bcc" TEXT,
    "ref" TEXT,
    "data" BLOB NULL,
    "created_at" INTEGER,
    "updated_at" INTEGER
);

CREATE TABLE IF NOT EXISTS users (
    "id" TEXT PRIMARY KEY,
    "type" TEXT,
    "flag" TEXT,
    "from" TEXT,
    "to" TEXT,
    "cc" TEXT,
    "bcc" TEXT,
    "ref" TEXT,
    "amount" REAL,
    "data" BLOB NULL,
    "created_at" INTEGER,
    "updated_at" INTEGER
);


CREATE TABLE IF NOT EXISTS crons (
    "id" TEXT PRIMARY KEY,
    "cc" TEXT,
    "bcc" TEXT,
    "job" BLOB NULL,
    "ref" TEXT,
    "created_at" INTEGER,
    "updated_at" INTEGER
);


CREATE TABLE IF NOT EXISTS console (
    "id" TEXT PRIMARY KEY,
    "bcc" TEXT,
    "log" TEXT,
    "created_at" INTEGER
);