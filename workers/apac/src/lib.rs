//! commerce-apac-logis-center — 원본 `apac/src/index.ts` 를 같은 동작으로 옮긴 것.
//!
//! 1분 cron 이 50초 동안 `Cron` 을 되풀이한다. `Cron` 은 지역 D1(`env[env.region]`)의 tasks 를 오래된 순으로 300개 읽고,
//! 하나씩 앞 작업이 끝났는지(팀 데이터가 바뀌었는지) 보고, 쓸 LLM 을 골라 lock 을 건 뒤 proxy 에 gzip JSON 으로 넘긴다.
//! 원본의 버릇(첫 행이 lock 이면 `ungzip(null)` 이 던져 그 회차가 멈추는 것 등)도 그대로다.

#[cfg(target_arch = "wasm32")]
mod worker_main {
    use logis_worker_common::js::{self, Body, Env};
    use logis_worker_common::jsv::Throw;
    use logis_worker_common::platform::{now, sleep};
    use logis_worker_common::tables::{lookup, LOGIS_REGION};
    use logis_worker_common::util::{is_diff, random_below};
    use logis_worker_common::{console_log, gzip, loose_eq, obj, stringify_str, R, V};
    use worker::{event, ScheduleContext, ScheduledEvent};

    const PROXY_URL: &str = "https://proxy.commerce.logis.center";

    /// `geminiKey(gemini1, gemini2)`
    fn gemini_key(g1: &V, g2: &V) -> V {
        if random_below(2) != 0 {
            obj! { "first" => g1.clone(), "second" => g2.clone() }
        } else {
            obj! { "first" => g2.clone(), "second" => g1.clone() }
        }
    }

    /// `geminiModel()` — 두 갈래가 같지만 난수는 원본처럼 한 번 뽑는다
    fn gemini_model() -> V {
        let _ = random_below(2);
        obj! { "first" => "gemini-flash-lite-latest", "second" => "gemini-flash-lite-latest" }
    }

    fn key2(a: &V, b: &V) -> String {
        format!("{}-{}", a.to_str(), b.to_str())
    }

    /// `Cron(event, env, ctx, models, limits, timeout)` → `{ length, models, limits }`
    async fn cron(env: &Env, models: V, limits: V, timeout: &V) -> V {
        let now = now();
        let created_at = now;
        let region = env.get("region").to_key();
        let mut len = V::Undef;
        let mut models = models;
        let mut limits = limits;

        let body: R<()> = async {
            let results = js::query_bound(
                env,
                &region,
                r#"SELECT * FROM tasks WHERE "created_at" < ?1 ORDER BY created_at ASC LIMIT 300"#,
                &[V::Num(created_at)],
            )
            .await?;
            len = results.length()?;
            let mut tasks = V::new_arr();

            if len.truthy() {
                console_log!("limits", stringify_str(&limits));
                console_log!("tasks len", len);

                let mut i = 0usize;
                while (i as f64) < len.to_num() {
                    let cron = results.gi(i)?;
                    // 고침: lock 행(data 가 null, created_at 1 이라 늘 맨 앞)을 풀기 전에 본다. 원본은 ungzip(null) 이 먼저 던져
                    // lock 이 걸린 동안 매 회차가 통째로 예외로 끝났다
                    if loose_eq(&cron.g("id")?, &V::from("lock")) {
                        tasks = V::new_arr();
                        break;
                    }
                    let task = gzip::ungzip_json(&cron.g("data")?)?;
                    if task.g("method")?.truthy() {
                        task.del("method")?;
                    }
                    tasks.push(task)?;
                    i += 1;
                }

                if tasks.len() > 0 {
                    let mut t = 0usize;
                    while t < tasks.len() {
                        let task = tasks.at(t);

                        sleep(timeout.g("value")?.to_num() * timeout.g("delay")?.to_num()).await;

                        let elapsed = now_ms() - timeout.g("start")?.to_num();
                        let time_left = timeout.g("max")?.to_num() - elapsed;
                        if time_left <= 5000.0 {
                            break;
                        }

                        let results = js::query_bound(
                            env,
                            &region,
                            r#"SELECT * FROM tasks WHERE "created_at" <= ?1 AND "updated_at" > 0 ORDER BY created_at ASC LIMIT 1"#,
                            &[task.g("created_at")?],
                        )
                        .await?;

                        if results.truthy() {
                            console_log!("results.length", results.length()?);

                            // `var _cron = tasks[t-1]` (t = 0 이면 tasks[-1] = undefined)
                            let mut cron_prev = if t == 0 { V::Undef } else { tasks.at(t - 1) };

                            if results.length()?.truthy() {
                                cron_prev = results.gi(0)?;
                                let prev_task = gzip::ungzip_json(&cron_prev.g("data")?)?;
                                let before = prev_task.g("team")?;
                                let logis_region = lookup(LOGIS_REGION, &prev_task.g("flag")?);
                                let users = js::query_bound(
                                    env,
                                    &logis_region.to_key(),
                                    r#"SELECT * FROM users WHERE "type" = 'team' AND "id" = ?1 AND "created_at" < ?2 LIMIT 1"#,
                                    &[before.g("id")?, V::Num(now)],
                                )
                                .await?;
                                let after = users.gi(0)?;
                                let data = gzip::ungzip_json(&after.g("data")?)?;
                                after.s_("data", data)?;

                                let diff = is_diff(&before.g("data")?, &after.g("data")?)?;
                                console_log!("diff", diff);

                                if diff {
                                    timeout.s_("delay", V::Num(timeout.g("delay")?.to_num() + 0.5))?;
                                    break;
                                } else {
                                    console_log!("1.remove 2.통과 진입");
                                    timeout.s_("delay", V::Num(0.5))?;
                                    js::run_bound(env, &region, r#"DELETE FROM tasks WHERE "id" = ?1"#, &[cron_prev.g("id")?]).await?;

                                    if loose_eq(&cron_prev.g("id")?, &task.g("id")?) {
                                        console_log!("self stop 진입");
                                        break;
                                    } else {
                                        limits.sk(&cron_prev.g("id")?, V::Bool(true))?;
                                        limits.s_("team", before)?;
                                    }
                                }
                            } else if cron_prev.truthy() {
                                if !limits.gk(&cron_prev.g("id")?)?.truthy() {
                                    timeout.s_("delay", V::Num(timeout.g("delay")?.to_num() + 0.5))?;
                                    console_log!("stop 진입");
                                    break;
                                } else {
                                    timeout.s_("delay", V::Num(1.0))?;
                                    console_log!("통과");
                                }
                            } else if t + 1 == tasks.len() && limits.get(&upper(&task.g("id")?)?).truthy() {
                                console_log!("미정");
                                timeout.s_("delay", V::Num(timeout.g("delay")?.to_num() + 0.5))?;
                                break;
                            }

                            let key = gemini_key(&env.get("gemini1"), &env.get("gemini2"));
                            let model = gemini_model();
                            let mut llm_api = V::from("");
                            let mut llm_model = V::from("");

                            let (kf, ks) = (key.get("first"), key.get("second"));
                            let (mf, ms) = (model.get("first"), model.get("second"));
                            if models.get(&key2(&kf, &mf)).truthy() {
                                llm_api = kf.clone();
                                llm_model = mf.clone();
                            } else if models.get(&key2(&kf, &ms)).truthy() {
                                llm_api = kf.clone();
                                llm_model = ms.clone();
                            } else if models.get(&key2(&ks, &mf)).truthy() {
                                llm_api = ks.clone();
                                llm_model = mf.clone();
                            } else if models.get(&key2(&ks, &ms)).truthy() {
                                llm_api = ks.clone();
                                llm_model = ms.clone();
                            } else if !models.get("deepinfra").truthy() {
                                // 고침: 원본은 id = "…"(큰따옴표 — SQLite 는 먼저 열 이름으로 읽는다)로 지웠다
                                js::run_bound(env, &region, r#"DELETE FROM tasks WHERE "id" = ?1"#, &[task.g("id")?]).await?;
                                t += 1;
                                continue;
                            }

                            // lock
                            let stmt = env.prepare(&region, LOCK_SQL)?;
                            let stmt = js::bind(
                                &stmt,
                                &[
                                    V::from("lock"),
                                    V::from(""),
                                    V::from(""),
                                    V::from(""),
                                    V::from(""),
                                    V::from(""),
                                    V::from(""),
                                    V::Null,
                                    V::Num(1.0),
                                    V::Num(0.0),
                                ],
                            )?;
                            js::run(&stmt).await?;

                            let payload = obj! {
                                "now" => now,
                                "id" => task.g("id")?,
                                "ref" => task.g("ref")?,
                                "region" => env.get("region"),
                                "models" => models.clone(),
                                "limits" => limits.clone(),
                                "gemini_llm_api" => llm_api,
                                "gemini_llm_model" => llm_model,
                                "deepinfra" => env.get("deepinfra"),
                            };
                            let arr = gzip::gzip_json(&payload)?;

                            let sent: R<()> = async {
                                let bytes = arr.to_bytes().unwrap_or_default();
                                let res = js::fetch(
                                    PROXY_URL,
                                    "POST",
                                    &[("Content-Type", "application/octet-stream"), ("Content-Encoding", "gzip")],
                                    Body::Bytes(bytes),
                                )
                                .await?;
                                let reply = js::response_json(&res).await?;
                                models = reply.g("models")?;
                                limits = reply.g("limits")?;
                                Ok(())
                            }
                            .await;
                            if let Err(err) = sent {
                                let stmt = env.prepare(&region, "\n\t\t\t\t\t\t\t\t\tDELETE FROM tasks WHERE \"id\" = 'lock'\n\t\t\t\t\t\t\t\t")?;
                                js::run(&stmt).await?;
                                console_log!("err", err);
                            }
                        }
                        t += 1;
                    }
                }
            } else {
                timeout.s_("delay", V::Num(timeout.g("delay")?.to_num() + 0.5))?;
            }
            Ok(())
        }
        .await;

        if let Err(err) = body {
            console_log!("batch err", err);
        }

        obj! { "length" => len, "models" => models, "limits" => limits }
    }

    fn now_ms() -> f64 {
        now()
    }

    /// `x.toUpperCase()` — 문자열이 아니면 원본처럼 던진다
    fn upper(v: &V) -> R<String> {
        match v {
            V::Str(s) => Ok(s.to_uppercase()),
            V::Undef | V::Null => Err(Throw::type_error(format!("Cannot read properties of {} (reading 'toUpperCase')", v.to_str()))),
            _ => Err(Throw::type_error("toUpperCase is not a function")),
        }
    }

    const LOCK_SQL: &str = "\n\t\t\t\t\t\t\tINSERT INTO tasks (\n\t\t\t\t\t\t\t\t\"id\", \"type\", \"from\", \"to\", \"cc\", \"bcc\", \"ref\", \"data\", \"created_at\", \"updated_at\"\n\t\t\t\t\t\t\t) VALUES (\n\t\t\t\t\t\t\t\t?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10 \n\t\t\t\t\t\t\t) ON CONFLICT (id) DO UPDATE SET\n\t\t\t\t\t\t\t\t\"type\" = EXCLUDED.\"type\",\n\t\t\t\t\t\t\t\t\"from\" = EXCLUDED.\"from\",\n\t\t\t\t\t\t\t\t\"to\" = EXCLUDED.\"to\",\n\t\t\t\t\t\t\t\t\"cc\" = EXCLUDED.\"cc\",\n\t\t\t\t\t\t\t\t\"bcc\" = EXCLUDED.\"bcc\",\n\t\t\t\t\t\t\t\t\"ref\" = EXCLUDED.\"ref\",\n\t\t\t\t\t\t\t\t\"data\" = EXCLUDED.\"data\",\n\t\t\t\t\t\t\t\t\"created_at\" = EXCLUDED.\"created_at\",\n\t\t\t\t\t\t\t\t\"updated_at\" = EXCLUDED.\"updated_at\"\n\t\t\t\t\t\t";

    #[event(scheduled)]
    pub async fn scheduled(_event: ScheduledEvent, env: worker::Env, _ctx: ScheduleContext) {
        logis_worker_common::platform::install_panic_hook();
        let env = Env(env);
        let limits = V::new_obj();
        let models = V::new_obj();

        models.put("deepinfra", V::Num(10000.0));
        models.put("cloudflare", V::Num(3000.0));
        let (g1, g2) = (env.get("gemini1").to_str(), env.get("gemini2").to_str());
        // 고침: 쿼터 키를 실제로 고르는 모델(gemini_model)에 맞춘다. 원본은 2.5 · 2.0 flash-lite 키만 채우고
        // gemini-flash-lite-latest 로 찾아, Gemini 가 한 번도 골라지지 않았다
        models.put(&format!("{g1}-gemini-flash-lite-latest"), V::Num(4000.0));
        models.put(&format!("{g2}-gemini-flash-lite-latest"), V::Num(4000.0));

        let timeout = obj! { "value" => 1000, "delay" => 0.5, "start" => now(), "max" => 50 * 1000 };

        let (mut limits, mut models) = (limits, models);
        loop {
            let elapsed = now() - timeout.get("start").to_num();
            let time_left = timeout.get("max").to_num() - elapsed;
            if time_left <= 5000.0 {
                break;
            }

            let results = cron(&env, models.clone(), limits.clone(), &timeout).await;
            limits = results.get("limits");
            models = results.get("models");

            if results.get("length").truthy() {
                timeout.put("delay", V::Num(0.5));
            } else {
                timeout.put("delay", V::Num(timeout.get("delay").to_num() + 0.5));
            }

            sleep(timeout.get("value").to_num() * timeout.get("delay").to_num()).await;
        }
    }
}
