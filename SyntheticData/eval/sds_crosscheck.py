"""
SDS 교차 확인
=============

앱이 AppData 에 남긴 SDS 실행 파일과 합성 데이터 정답을 맞대어 봅니다.

  python sds_crosscheck.py pages <평가 폴더> <결과.jsonl> <실행 로그>     페이지 판정과 commerce.* 축
  python sds_crosscheck.py fields <평가 폴더> <결과.jsonl>               필드 배정 · 혼동 사전과 저장 값
  python sds_crosscheck.py search <평가 폴더> <결과.jsonl> <채점 상세.json>  질의 성공과 search.* 축
  python sds_crosscheck.py ledger <SDS 폴더>                             원장과 실행 파일이 같은 사실을 말하는지

SDS 는 정답을 모르는 채로 쌓입니다. 정답을 아는 합성 데이터에서는 그 축이 실제로 맞고 틀림을 가르는지 잴 수 있습니다.
AUC 0.5 는 아무 정보가 없다는 뜻이고, 1.0 이나 0.0 에 가까울수록 축이 판정 오류를 미리 알려 줍니다.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from score_commerce import check_field, read_jsonl, status_table
from score_pages import parse_log
from score_vision import auc
from sds_runs import attach_runs, axes_of, confusions_of, decay_of, fields_of, ledger_root, load_runs
from truth import commerce_pages


def cmd_pages(a) -> int:
    results = read_jsonl(a.results)
    runs = load_runs(a.root)
    by_id = attach_runs(results, runs)
    log = parse_log(a.log)
    truth = {p["page_id"]: p for p in commerce_pages()}
    rows = []
    for r in results:
        p = truth.get(r["id"])
        got = log.get(r["id"])
        if p is None or got is None or got.get("layout") is None:
            continue
        axes, margins = {}, {}
        for run in by_id.get(r["id"], []):
            axes.update({k: v[1] for k, v in axes_of(run).items()})
            margins.update(decay_of(run))
        rows.append({
            "id": r["id"], "layout_ok": got["layout"] == p["layout"], "type_ok": got.get("page_type") == p["page_type"],
            "axes": axes, "margins": margins,
        })
    print(f"페이지 {len(rows)} | 실행 파일 {len(runs)} | 레이아웃 맞음 {sum(x['layout_ok'] for x in rows)} | 유형 맞음 {sum(x['type_ok'] for x in rows)}")
    names = Counter()
    for x in rows:
        names.update(x["axes"].keys())
        names.update("margin:" + k for k in x["margins"].keys())
    table = []
    for name, cnt in names.items():
        if cnt < len(rows) * 0.5:
            continue
        def val(x):
            return x["margins"].get(name[7:]) if name.startswith("margin:") else x["axes"].get(name)
        for label in ("layout_ok", "type_ok"):
            pos = [val(x) for x in rows if x[label] and val(x) is not None]
            neg = [val(x) for x in rows if not x[label] and val(x) is not None]
            if len(pos) >= 3 and len(neg) >= 3:
                table.append((abs(auc(pos, neg) - 0.5), auc(pos, neg), label, name, len(pos), len(neg)))
    table.sort(reverse=True)
    print(f"{'AUC':>6s} {'판정':10s} {'맞음':>4s} {'틀림':>4s} 축")
    for _, value, label, name, np, nn in table[:30]:
        print(f"{value:6.3f} {label:10s} {np:4d} {nn:4d} {name}")
    return 0


def cmd_fields(a) -> int:
    results = read_jsonl(a.results)
    runs = load_runs(a.root)
    by_id = attach_runs(results, runs)
    truth = {p["page_id"]: p for p in commerce_pages()}
    statuses = status_table()
    per_field = defaultdict(Counter)
    conf = Counter()
    for r in results:
        p = truth.get(r["id"])
        if p is None:
            continue
        sds_fields, sds_conf = {}, {}
        for run in by_id.get(r["id"], []):
            sds_fields.update(fields_of(run))
            for k, v in confusions_of(run).items():
                sds_conf[k] = v
        for k, v in sds_conf.items():
            conf[k] += v["ties"]
        rows = p["items"] if p["layout"] == "list" else [p["fields"]]
        for row in rows:
            rid = str(row.get("id,link"))
            item = next((i for i in r.get("items") or [] if rid in (str((i.get("data") or {}).get("link")) + (i.get("text") or ""))), None)
            for field, v in row.items():
                if field == "id,link":
                    continue
                verdict = check_field(field, v, (item or {}).get("data") or {}, statuses) if item else "없음"
                c = per_field[(p["page_type"], field)]
                c[verdict] += 1
                c["sds_assigned"] += sds_fields.get(field, {}).get("assigned", 0)
                c["sds_near"] += sds_fields.get(field, {}).get("near_miss", 0)
    print(f"{'유형':9s} {'필드':22s} {'맞음':>5s} {'틀림':>5s} {'없음':>5s} {'SDS배정':>7s} {'니어미스':>7s}")
    for (ptype, field), c in sorted(per_field.items(), key=lambda kv: -kv[1]["틀림"]):
        print(f"{ptype:9s} {field:22s} {c['맞음']:5d} {c['틀림']:5d} {c['없음']:5d} {c['sds_assigned']:7d} {c['sds_near']:7d}")
    print("혼동 사전 상위", conf.most_common(12))
    return 0


def cmd_search(a) -> int:
    """score_search.py --dump 의 질의별 성공을 search.* 축과 맞댑니다. 첫 정답이 10위 안이면 성공입니다."""
    results = read_jsonl(a.results)
    detail = json.loads(a.detail.read_text(encoding="utf-8"))["detail"]
    by_id = attach_runs(results, load_runs(a.root))
    rows = []
    for r, d in zip(results, detail):
        axes = {}
        for run in by_id.get(r["id"], []):
            axes.update({k: v[1] for k, v in axes_of(run).items()})
        rows.append({"ok": d["mrr"] >= 0.1, "axes": axes, "kind": d["kind"]})
    names = Counter()
    for x in rows:
        names.update(x["axes"].keys())
    print(f"질의 {len(rows)} | 성공 {sum(x['ok'] for x in rows)} | 축 {len(names)}")
    table = []
    for name, cnt in names.items():
        pos = [x["axes"][name] for x in rows if x["ok"] and name in x["axes"]]
        neg = [x["axes"][name] for x in rows if not x["ok"] and name in x["axes"]]
        if len(pos) >= 3 and len(neg) >= 3:
            value = auc(pos, neg)
            table.append((abs(value - 0.5), value, name, len(pos), len(neg)))
    table.sort(reverse=True)
    print(f"{'AUC':>6s} {'성공':>4s} {'실패':>4s} 축")
    for _, value, name, np, nn in table[:24]:
        print(f"{value:6.3f} {np:4d} {nn:4d} {name}")
    return 0


def cmd_ledger(a) -> int:
    """원장의 필드 관측 합과 실행 파일의 합을 비교합니다. 원장은 실행 파일의 누적이어야 합니다."""
    root = a.root
    for ledger in ("extract", "index", "search"):
        f = root / f"{ledger}.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        ledger_seen = sum(x.get("seen", 0) for s in data["scopes"].values() for x in s.get("field", {}).values())
        ledger_base = sum(w.get("n", 0) for s in data["scopes"].values() for w in s.get("baseline", {}).values())
        run_seen = run_base = files = 0
        for rf in sorted((root / "runs" / ledger).glob("*.json")) if (root / "runs" / ledger).exists() else []:
            d = json.loads(rf.read_text(encoding="utf-8"))
            files += 1
            run_seen += sum(x.get("seen", 0) for s in d["scopes"].values() for x in s.get("field", {}).values())
            run_base += sum(w.get("n", 0) for s in d["scopes"].values() for w in s.get("baseline", {}).values())
        print(f"{ledger:8s} 실행 파일 {files:3d} | 필드 관측 원장 {ledger_seen:6d} / 실행 합 {run_seen:6d} | 기준선 원장 {ledger_base:6d} / 실행 합 {run_base:6d} | 팀 {data.get('team', '')[:10]!r}")
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pages")
    p.add_argument("root", type=Path)
    p.add_argument("results", type=Path)
    p.add_argument("log", type=Path)
    p = sub.add_parser("fields")
    p.add_argument("root", type=Path)
    p.add_argument("results", type=Path)
    p = sub.add_parser("search")
    p.add_argument("root", type=Path)
    p.add_argument("results", type=Path)
    p.add_argument("detail", type=Path)
    p = sub.add_parser("ledger")
    p.add_argument("root", type=Path)
    a = ap.parse_args()
    return {"pages": cmd_pages, "fields": cmd_fields, "search": cmd_search, "ledger": cmd_ledger}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
