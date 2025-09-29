#!/usr/bin/env python3
import os, sys, io, re, zipfile, json
from statistics import mean
import requests

GITHUB_API = "https://api.github.com"
IGNORE_KEYS = {"compile_time", "runtime_fps", "realtime_factor"}

# ---------- parsing ----------
def parse_speed_txt_lines(lines):
    """
    Parse speed_test.txt lines into a map of { test_id: {"runtime_fps": float|None, "compile_time": float|None} }
    """
    out = {}
    for line in lines:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.strip().split('|') if '=' in p]
        kv = {}
        for p in parts:
            k, v = p.split('=', 1)
            kv[k.strip()] = v.strip()
        # test_id construction (exclude keys that are not needed for comparison, sort)
        items = [(k, kv[k]) for k in kv.keys() if k not in IGNORE_KEYS]
        items.sort()
        test_id = '|'.join(f'{k}={v}' for k, v in items)
        # parse values
        rt = kv.get('runtime_fps')
        ct = kv.get('compile_time')
        try:
            rt = float(rt) if rt is not None else None
        except: rt = None
        try:
            ct = float(ct) if ct is not None else None
        except: ct = None
        out[test_id] = {"runtime_fps": rt, "compile_time": ct}
    return out

def parse_speed_txt_from_zip_bytes(zip_bytes):
    """Parse speed_test*.txt files from artifact zip and merge into a dict"""
    merged = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if re.search(r"speed_test.*\.txt$", name):
                with zf.open(name) as f:
                    lines = io.TextIOWrapper(f, encoding='utf-8', errors='ignore').read().splitlines()
                    cur = parse_speed_txt_lines(lines)
                    merged.update(cur)
    return merged

# ---------- GitHub helpers ----------
def gh_json(session, url, params=None, headers=None):
    r = session.get(url, params=params, headers=headers or {})
    if r.status_code != 200:
        raise RuntimeError(f"GET {url} failed: {r.status_code} {r.text}")
    return r.json()

def gh_bin(session, url):
    # Accept is not forced: API redirects to zip
    r = session.get(url, allow_redirects=True)
    if r.status_code == 415:
        r = session.get(url, headers={"Accept": "application/vnd.github+json"}, allow_redirects=True)
    ct = (r.headers.get("Content-Type") or "").lower()
    if r.status_code == 200 and ("zip" in ct or "octet-stream" in ct or "binary" in ct):
        return r.content
    try:
        j = r.json()
        raise RuntimeError(f"GET(bin) {url} failed: {r.status_code} {j}")
    except ValueError:
        raise RuntimeError(f"GET(bin) {url} failed: {r.status_code} {r.text[:300]}")

# ---------- main ----------
def main():
    # Required/optional ENV
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    pr_number = os.environ.get("PR_NUMBER")  # ${{ github.event.pull_request.number }}
    workflow_name = os.environ.get("WORKFLOW_NAME", "Production")

    # Regression threshold (%) — runtime is regression if decrease compared to mean is greater than threshold, 
    # compile is regression if increase is greater than threshold
    tol_runtime = float(os.environ.get("RUNTIME_REGRESSION_TOLERANCE_PCT", "10"))
    tol_compile = float(os.environ.get("COMPILE_REGRESSION_TOLERANCE_PCT", "10"))

    baselines_k = int(os.environ.get("REGRESSION_BASELINES", "5"))
    current_txt_path = os.environ.get("CURRENT_SPEED_TXT_PATH")

    # Debug output files (for artifact upload)
    out_json_path = os.environ.get("OUTPUT_DEBUG_JSON", "benchmark_compare_result.json")

    # 1) Fail if token is missing (policy)
    if not token:
        print("Missing GITHUB_TOKEN (or GH_TOKEN). Failing by policy.", file=sys.stderr)
        sys.exit(2)

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "Accept": "application/vnd.github+json",
    })

    # 2) Parse speed_test.txt of current PR run (local file first)
    current_map = {}
    if current_txt_path and os.path.exists(current_txt_path):
        with open(current_txt_path, "r", encoding="utf-8", errors="ignore") as f:
            current_map = parse_speed_txt_lines(f.read().splitlines())
    else:
        if not (repo and run_id):
            print("No CURRENT_SPEED_TXT_PATH and missing GITHUB_REPOSITORY/GITHUB_RUN_ID.", file=sys.stderr)
            sys.exit(2)
        owner, repo_name = repo.split("/", 1)
        arts = gh_json(session, f"{GITHUB_API}/repos/{owner}/{repo_name}/actions/runs/{run_id}/artifacts")
        for a in arts.get("artifacts", []):
            if a.get("name") == "speed-test-results" and not a.get("expired", False):
                zip_bytes = gh_bin(session, a["archive_download_url"])
                current_map = parse_speed_txt_from_zip_bytes(zip_bytes)
                break

    if not current_map:
        print("No current speed_test data found.", file=sys.stderr)
        sys.exit(2)

    # 3) Collect baselines from recent N runs of main branch in the same workflow
    if not repo:
        print("Missing GITHUB_REPOSITORY.", file=sys.stderr)
        sys.exit(2)
    owner, repo_name = repo.split("/", 1)

    # Workflow ID
    wfs = gh_json(session, f"{GITHUB_API}/repos/{owner}/{repo_name}/actions/workflows")
    wf_id = None
    for wf in wfs.get("workflows", []):
        if wf.get("name") == workflow_name:
            wf_id = wf["id"]; break
    if not wf_id:
        print(f"Workflow '{workflow_name}' not found.", file=sys.stderr)
        sys.exit(2)

    baseline_rt = {}  # test_id -> [runtime_fps...]
    baseline_ct = {}  # test_id -> [compile_time...]
    page = 1
    collected = 0
    while collected < baselines_k and page <= 10:
        runs = gh_json(session, f"{GITHUB_API}/repos/{owner}/{repo_name}/actions/workflows/{wf_id}/runs",
                       params={"branch": "main", "status": "success", "per_page": 50, "page": page})
        page += 1
        for r in runs.get("workflow_runs", []):
            if str(r.get("id")) == str(run_id):
                continue
            r_arts = gh_json(session, r["artifacts_url"])
            got = False
            for a in r_arts.get("artifacts", []):
                if a.get("name") == "speed-test-results" and not a.get("expired", False):
                    zip_bytes = gh_bin(session, a["archive_download_url"])
                    base_map = parse_speed_txt_from_zip_bytes(zip_bytes)
                    for tid, vals in base_map.items():
                        if (v := vals.get("runtime_fps")) is not None:
                            baseline_rt.setdefault(tid, []).append(float(v))
                        if (c := vals.get("compile_time")) is not None:
                            baseline_ct.setdefault(tid, []).append(float(c))
                    got = True
                    break
            if got:
                collected += 1
                if collected >= baselines_k:
                    break

    # 4) Compare (runtime_fps ↓, compile_time ↑ each based on mean)
    runtime_regs = []  # (tid, curr, base_mean, delta_pct)
    compile_regs = []  # (tid, curr, base_mean, delta_pct)
    results = []       # per test combined result

    for tid, vals in current_map.items():
        curr_rt = vals.get("runtime_fps")
        curr_ct = vals.get("compile_time")

        base_rt_mean = None
        base_ct_mean = None
        d_rt = None
        d_ct = None
        st_rt = "no-baseline"
        st_ct = "no-baseline"

        # runtime comparison (baseline mean)
        bases_rt = baseline_rt.get(tid, [])
        if bases_rt and curr_rt is not None:
            base_rt_mean = mean(bases_rt)
            if base_rt_mean > 0:
                d_rt = (curr_rt - base_rt_mean) / base_rt_mean * 100.0
                st_rt = "ok" if d_rt >= -tol_runtime else "regressed"
                if st_rt == "regressed":
                    runtime_regs.append((tid, curr_rt, base_rt_mean, d_rt))

        # compile comparison (baseline mean)
        bases_ct = baseline_ct.get(tid, [])
        if bases_ct and curr_ct is not None:
            base_ct_mean = mean(bases_ct)
            if base_ct_mean > 0:
                d_ct = (curr_ct - base_ct_mean) / base_ct_mean * 100.0
                st_ct = "ok" if d_ct <= tol_compile else "regressed"
                if st_ct == "regressed":
                    compile_regs.append((tid, curr_ct, base_ct_mean, d_ct))

        results.append({
            "id": tid,
            "runtime": {"current": curr_rt, "baseline_mean": base_rt_mean, "delta_pct": d_rt, "status": st_rt},
            "compile": {"current": curr_ct, "baseline_mean": base_ct_mean, "delta_pct": d_ct, "status": st_ct},
        })

    # 5) Create comment body (if there is a regression, comment)
    comment_body = None
    if pr_number and (runtime_regs or compile_regs):
        def trunc(s, n=140): return (s[:n] + "…") if len(s) > n else s

        lines = []
        lines.append(":warning: **Benchmark regression detected**")
        lines.append("")
        lines.append(f"- Runtime tolerance: **-{tol_runtime:.1f}%** (slower than baseline mean)")
        lines.append(f"- Compile tolerance: **+{tol_compile:.1f}%** (longer than baseline mean)")
        lines.append(f"- Baselines used: **{min(collected, baselines_k)}** runs from `main`")
        lines.append("")

        if runtime_regs:
            # First the most degraded (more negative)
            runtime_regs.sort(key=lambda x: x[3])  
            lines.append("**Runtime FPS regressions (vs mean)**")
            lines.append("")
            lines.append("| test_id (truncated) | current FPS | baseline mean | delta % |")
            lines.append("|---|---:|---:|---:|")
            for tid, curr, base, d in runtime_regs[:20]:
                lines.append(f"| `{trunc(tid)}` | {curr:,.0f} | {base:,.0f} | {d:.2f}% |")
            if len(runtime_regs) > 20:
                lines.append("_Only first 20 shown._")
            lines.append("")

        if compile_regs:
            # First the most degraded (more positive)
            compile_regs.sort(key=lambda x: -x[3])  
            lines.append("**Compile-time regressions (vs mean)**")
            lines.append("")
            lines.append("| test_id (truncated) | current time | baseline mean | delta % |")
            lines.append("|---|---:|---:|---:|")
            for tid, curr, base, d in compile_regs[:20]:
                lines.append(f"| `{trunc(tid)}` | {curr:,.0f} | {base:,.0f} | {d:.2f}% |")
            if len(compile_regs) > 20:
                lines.append("_Only first 20 shown._")
            lines.append("")

        comment_body = "\n".join(lines)

        # Send PR comment
        url = f"{GITHUB_API}/repos/{owner}/{repo_name}/issues/{pr_number}/comments"
        resp = session.post(url, json={"body": comment_body})
        if resp.status_code not in (200, 201):
            print(f"Failed to post PR comment: {resp.status_code} {resp.text}", file=sys.stderr)

    # 6) Save debug output files
    debug_obj = {
        "tolerance_runtime_pct": tol_runtime,
        "tolerance_compile_pct": tol_compile,
        "baselines_used": min(collected, baselines_k),
        "counts": {
            "tests": len(results),
            "runtime_regressions": len(runtime_regs),
            "compile_regressions": len(compile_regs),
        },
        "results": results,
    }
    try:
        with open(out_json_path, "w") as f:
            json.dump(debug_obj, f, indent=2)
    except Exception as e:
        print(f"Failed to write {out_json_path}: {e}", file=sys.stderr)

    # Exit normally regardless of regression
    sys.exit(0)

if __name__ == "__main__":
    main()
