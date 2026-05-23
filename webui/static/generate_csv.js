/* generate_csv.js — client-side logic for the Generate Test CSV page */
(function () {
    "use strict";

    // ---------------------------------------------------------------------------
    // Name generation helpers
    // ---------------------------------------------------------------------------
    const FIRST_NAMES = [
        "Aaron","Abigail","Adam","Adrian","Aisha","Alex","Alicia","Aliyah","Amanda","Amber",
        "Amelia","Andre","Andrew","Angela","Ann","Anthony","Ashley","Ayesha","Barbara","Benjamin",
        "Brandon","Brianna","Caleb","Cameron","Carlos","Carmen","Chantel","Chelsea","Christian","Christine",
        "Christopher","Cindy","Claire","Clayton","Cody","Crystal","Damian","Daniel","Danielle","David",
        "Diana","Dominic","Dylan","Eduardo","Elena","Elizabeth","Emily","Emma","Eric","Ethan",
        "Faith","Fatima","Fernando","Frank","Gabriel","George","Grace","Hannah","Hector","Henry",
        "Imani","Isaiah","Jacob","Jade","James","Janet","Jasmine","Jason","Jennifer","Jessica",
        "Joel","Jonathan","Jordan","Jose","Joshua","Juan","Julian","Kayla","Kevin","Kiran",
        "Kyle","Laura","Lauren","Leah","Leonardo","Leslie","Liam","Lisa","Logan","Lucas",
        "Luis","Madison","Marcus","Maria","Mason","Matthew","Maya","Mia","Michael","Michelle",
        "Miguel","Mira","Nathan","Natalie","Nicholas","Nicole","Noah","Nora","Olivia","Omar",
        "Patrick","Paul","Peter","Rachel","Rebecca","Richard","Robert","Ryan","Samantha","Samuel",
        "Sandra","Sara","Sarah","Shawn","Sofia","Stephanie","Steven","Susan","Taylor","Thomas",
        "Timothy","Tyler","Victoria","Vincent","Whitney","William","Yasmine","Zachary","Zoe","Zara"
    ];
    const LAST_NAMES = [
        "Adams","Alexander","Allen","Anderson","Andrews","Archer","Armstrong","Atkins","Austin","Bailey",
        "Baker","Banks","Barnes","Bell","Bennett","Bishop","Black","Blake","Boyd","Brooks",
        "Brown","Bryan","Burke","Burns","Butler","Campbell","Carr","Carter","Chambers","Chapman",
        "Charles","Clarke","Coleman","Collins","Cook","Cooper","Cox","Crawford","Cruz","Davis",
        "Dean","Dixon","Edwards","Ellis","Evans","Ferguson","Fisher","Fleming","Fletcher","Ford",
        "Foster","Fox","Francis","Fraser","Freeman","Garcia","Gibson","Gill","Gordon","Graham",
        "Grant","Gray","Green","Griffin","Hall","Hamilton","Harris","Harrison","Hart","Harvey",
        "Hayes","Henderson","Henry","Hill","Holmes","Howard","Hughes","Hunter","Jackson","James",
        "Jenkins","Johnson","Jones","Joseph","Kelly","Kennedy","King","Knight","Lambert","Lawrence",
        "Lewis","Long","Lopez","Martin","Martinez","Mason","Mathews","Mitchell","Moore","Morgan",
        "Morris","Morrison","Murray","Nelson","Newton","Nichols","Noel","Oliver","Owens","Palmer",
        "Parker","Patterson","Payne","Perry","Peters","Phillips","Pierre","Porter","Powell","Price",
        "Ramkissoon","Reid","Richards","Richardson","Roberts","Robinson","Rogers","Ross","Russell","Sanchez",
        "Sanders","Scott","Shaw","Singh","Smith","Spencer","Stewart","Sullivan","Taylor","Thomas",
        "Thompson","Torres","Turner","Walker","Ward","Watson","White","Williams","Wilson","Wood"
    ];

    const SETTINGS_URL = "/api/v1/settings";
    const GENERATE_URL = "/api/v1/generate-csv";
    const FALLBACK_MAX_ROWS = 40000;
    const NAME_COMBINATIONS = FIRST_NAMES.length * LAST_NAMES.length;
    // Prime and coprime with NAME_COMBINATIONS, so we walk every pair before
    // repeating while still looking random to a user scanning the CSV.
    const NAME_STEP = 9973;
    let maxRows = FALLBACK_MAX_ROWS;
    let pdfMaxRows = 20000;
    let zipMaxRows = FALLBACK_MAX_ROWS;

    // ---------------------------------------------------------------------------
    // Deterministic preview helpers
    // ---------------------------------------------------------------------------

    function hashString(value) {
        let hash = 2166136261;
        for (let i = 0; i < value.length; i++) {
            hash ^= value.charCodeAt(i);
            hash = Math.imul(hash, 16777619);
        }
        return hash >>> 0;
    }

    function realisticName(rowIndex, seed) {
        const zeroIndex = rowIndex - 1;
        const combo = (seed + (zeroIndex * NAME_STEP)) % NAME_COMBINATIONS;
        const first = FIRST_NAMES[combo % FIRST_NAMES.length];
        const last = LAST_NAMES[Math.floor(combo / FIRST_NAMES.length) % LAST_NAMES.length];
        if (zeroIndex < NAME_COMBINATIONS) return `${first} ${last}`;
        return `${first} ${last} ${Math.floor(zeroIndex / NAME_COMBINATIONS) + 1}`;
    }

    function rowValues(rowIndex, schoolName, examName, candidateStartNum, nameStyle, nameSeed) {
        const name = nameStyle === "random"
            ? realisticName(rowIndex, nameSeed)
            : `Student ${rowIndex}`;
        const cand = String(candidateStartNum + BigInt(rowIndex - 1)).padStart(10, "0");
        const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");
        return [name, schoolName, examName, cand, `${slug}.png`];
    }

    function escapeCsv(value) {
        const s = String(value ?? "");
        if (s.includes(",") || s.includes('"') || s.includes("\n") || s.includes("\r")) {
            return `"${s.replace(/"/g, '""')}"`;
        }
        return s;
    }

    // ---------------------------------------------------------------------------
    // Preview rendering
    // ---------------------------------------------------------------------------

    const PREVIEW_ROWS = 5;

    function renderPreview(count, schoolName, examName, candidateStart, nameStyle) {
        const previewEl = document.getElementById("gen-preview");
        const labelEl = document.getElementById("preview-label");
        const tbody = document.getElementById("preview-body");

        const candidateStartNum = BigInt(candidateStart);
        const nameSeed = hashString(`${schoolName}|${examName}|${candidateStart}`);

        tbody.innerHTML = "";
        const shown = Math.min(count, PREVIEW_ROWS);
        for (let i = 1; i <= shown; i++) {
            const tr = document.createElement("tr");
            rowValues(i, schoolName, examName, candidateStartNum, nameStyle, nameSeed).forEach(val => {
                const td = document.createElement("td");
                td.textContent = val;
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        }

        if (count > PREVIEW_ROWS) {
            const tr = document.createElement("tr");
            const td = document.createElement("td");
            td.colSpan = 5;
            td.className = "muted small";
            td.style.textAlign = "center";
            td.textContent = `… and ${count - PREVIEW_ROWS} more rows`;
            tr.appendChild(td);
            tbody.appendChild(tr);
        }

        labelEl.textContent = `(first ${Math.min(count, PREVIEW_ROWS)} of ${count} rows)`;
        previewEl.style.display = "";
    }

    // ---------------------------------------------------------------------------
    // Validation helpers
    // ---------------------------------------------------------------------------

    function showError(msg) {
        const el = document.getElementById("gen-error");
        el.textContent = msg;
        el.style.display = msg ? "" : "none";
    }

    function hasDesktopApi() {
        return Boolean(window.pywebview && window.pywebview.api);
    }

    async function saveDesktopUrl(downloadUrl, filename) {
        const result = await window.pywebview.api.save_download_url(downloadUrl, filename || "test_students.csv");
        if (!result.ok && !result.cancelled) {
            throw new Error(result.message || "Download failed.");
        }
    }

    function setLoading(button, message) {
        button.disabled = true;
        button.textContent = message;
    }

    function clearLoading(button) {
        button.disabled = false;
        button.textContent = "Generate & Download CSV";
    }

    async function loadRuntimeLimits() {
        const hint = document.getElementById("gen-limit-hint");
        const countInput = document.getElementById("g-count");
        try {
            const response = await fetch(SETTINGS_URL, { cache: "no-store" });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const settings = await response.json();
            pdfMaxRows = Number(settings.prefill_pdf_max_rows) || pdfMaxRows;
            zipMaxRows = Number(settings.prefill_zip_max_rows) || zipMaxRows;
            maxRows = zipMaxRows;
            countInput.max = String(maxRows);
            hint.textContent = `Current defaults: PDF prefill accepts ${pdfMaxRows.toLocaleString()} rows; ZIP prefill and test CSV accept ${zipMaxRows.toLocaleString()} rows. Change these on Settings.`;
        } catch (err) {
            maxRows = FALLBACK_MAX_ROWS;
            countInput.max = String(maxRows);
            hint.textContent = "Using default limits: PDF prefill accepts 20,000 rows; ZIP prefill and test CSV accept 40,000 rows.";
        }
    }

    function getFormValues() {
        const count = parseInt(document.getElementById("g-count").value, 10);
        const schoolName = document.getElementById("g-school").value.trim();
        const examName = document.getElementById("g-exam").value.trim();
        const candidateStart = document.getElementById("g-start").value.trim();
        const nameStyle = document.querySelector('input[name="name_style"]:checked')?.value ?? "numbered";

        const errors = [];
        if (!Number.isInteger(count) || count < 1 || count > maxRows) {
            errors.push(`Number of students must be between 1 and ${maxRows.toLocaleString()}.`);
        }
        if (!schoolName) errors.push("School name is required.");
        if (!examName) errors.push("Exam name is required.");
        if (!/^\d{10}$/.test(candidateStart)) {
            errors.push("Candidate number start must be exactly 10 digits.");
        } else if (Number.isInteger(count) && count > 0) {
            const lastCandidate = BigInt(candidateStart) + BigInt(count) - 1n;
            if (lastCandidate > 9999999999n) {
                errors.push("Candidate numbers would exceed 10 digits. Lower the row count or use a smaller Candidate Number Start.");
            }
        }
        return { count, schoolName, examName, candidateStart, nameStyle, errors };
    }

    async function requestCsvDownload(values) {
        const formData = new FormData();
        formData.append("count", String(values.count));
        formData.append("school_name", values.schoolName);
        formData.append("exam_name", values.examName);
        formData.append("candidate_start", values.candidateStart);
        formData.append("name_style", values.nameStyle);

        const response = await fetch(GENERATE_URL, {
            method: "POST",
            body: formData,
            cache: "no-store"
        });
        if (!response.ok) {
            let detail = `Server error ${response.status}`;
            try {
                const body = await response.json();
                detail = body.detail || detail;
            } catch (_) {}
            throw new Error(detail);
        }
        return response.json();
    }

    // ---------------------------------------------------------------------------
    // Wire up form
    // ---------------------------------------------------------------------------

    document.addEventListener("DOMContentLoaded", () => {
        const form = document.getElementById("gen-form");
        const submitBtn = document.getElementById("gen-submit");
        loadRuntimeLimits();

        form.addEventListener("submit", (e) => {
            e.preventDefault();
        });

        submitBtn.addEventListener("click", async () => {
            showError("");

            const values = getFormValues();
            const { count, schoolName, examName, candidateStart, nameStyle, errors } = values;
            if (errors.length) {
                showError(errors.join(" "));
                return;
            }

            try {
                renderPreview(count, schoolName, examName, candidateStart, nameStyle);
                setLoading(submitBtn, `Preparing ${count.toLocaleString()} rows on server…`);
                const payload = await requestCsvDownload(values);
                setLoading(submitBtn, "Opening download…");
                if (hasDesktopApi()) {
                    await saveDesktopUrl(payload.download_url, payload.filename);
                    return;
                }
                window.location.href = payload.download_url;
            } catch (err) {
                showError(`Failed to generate CSV: ${err.message}`);
            } finally {
                clearLoading(submitBtn);
            }
        });
    });
})();
