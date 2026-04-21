const panel = document.querySelector("[data-batch-id]")
const batchId = panel ? panel.dataset.batchId : null
const STATUS_CLASSES = ["status-created", "status-queued", "status-running", "status-done", "status-failed"]

const show = (element, message, kind = "info") => {
    if (!element) return
    element.hidden = false
    element.textContent = message
    element.classList.remove("error", "success")
    if (kind === "error") element.classList.add("error")
    if (kind === "success") element.classList.add("success")
}

const apiUrl = (path) => `/api/v1/batches/${batchId}${path}`

const jsonFetch = async (url, options = {}) => {
    const response = await fetch(url, options)
    const text = await response.text()
    const data = text ? JSON.parse(text) : {}
    if (!response.ok) {
        throw new Error(data.detail || `Request failed (${response.status})`)
    }
    return data
}

const updateStatus = (status, lastError) => {
    const pill = document.getElementById("status-pill")
    if (pill) {
        STATUS_CLASSES.forEach((cls) => pill.classList.remove(cls))
        pill.classList.add(`status-${status}`)
        pill.textContent = status
    }
    const errorEl = document.getElementById("last-error")
    if (errorEl) {
        if (lastError) {
            errorEl.hidden = false
            errorEl.textContent = lastError
        } else {
            errorEl.hidden = true
            errorEl.textContent = ""
        }
    }
    const runBtn = document.getElementById("run-btn")
    if (runBtn) {
        runBtn.disabled = status === "queued" || status === "running"
        runBtn.textContent = status === "running" ? "Running..." : "Run OMR"
    }
}

const pollStatus = async () => {
    try {
        const data = await jsonFetch(apiUrl("/status"))
        updateStatus(data.status, data.last_error)
        if (data.status === "queued" || data.status === "running") {
            setTimeout(pollStatus, 2000)
        } else if (data.status === "done") {
            await refreshResults()
        }
    } catch (error) {
        console.error(error)
    }
}

const refreshFiles = async () => {
    try {
        const files = await jsonFetch(apiUrl("/files"))
        const list = document.getElementById("file-list")
        const count = document.getElementById("file-count")
        const empty = document.getElementById("no-files-msg")
        if (count) count.textContent = `(${files.length})`
        if (!list && files.length) {
            window.location.reload()
            return
        }
        if (!list) return
        list.innerHTML = ""
        files.forEach((file) => {
            const li = document.createElement("li")
            const sizeKb = (file.size_bytes / 1024).toFixed(1)
            li.innerHTML = `
                <span class="mono"></span>
                <span class="muted small">${sizeKb} KB</span>
                <button class="btn danger small" type="button" data-delete-file=""></button>
            `
            li.querySelector(".mono").textContent = file.name
            const button = li.querySelector("button")
            button.dataset.deleteFile = file.name
            button.textContent = "Remove"
            list.appendChild(li)
        })
        if (empty) empty.hidden = files.length > 0
    } catch (error) {
        console.error(error)
    }
}

const refreshResults = async () => {
    try {
        const data = await jsonFetch(apiUrl("/results"))
        const container = document.getElementById("results-container")
        if (!container) return
        if (!data.rows || data.rows.length === 0) {
            container.innerHTML = '<p class="muted" id="no-results-msg">No results yet. Run the batch to generate results.</p>'
            return
        }
        const columns = data.columns
        const header = columns.map((col) => `<th>${escapeHtml(col)}</th>`).join("")
        const rows = data.rows.map((row) => {
            const cells = columns.map((col) => {
                if (col === "file_id") return `<td class="mono">${escapeHtml(row.file_id || "")}</td>`
                if (col === "input_path") return `<td class="mono small">${escapeHtml(row.input_path || "")}</td>`
                if (col === "output_path") return `<td class="mono small">${escapeHtml(row.output_path || "")}</td>`
                if (col === "score") return `<td>${escapeHtml(row.score || "")}</td>`
                return `<td>${escapeHtml((row.responses && row.responses[col]) || "")}</td>`
            }).join("")
            return `<tr>${cells}</tr>`
        }).join("")
        container.innerHTML = `
            <table class="table">
                <thead><tr>${header}</tr></thead>
                <tbody>${rows}</tbody>
            </table>
        `
    } catch (error) {
        console.error(error)
    }
}

const escapeHtml = (value) => {
    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;")
}

const handleUpload = async (event) => {
    event.preventDefault()
    const feedback = document.getElementById("upload-feedback")
    const input = document.getElementById("file-input")
    if (!input.files || input.files.length === 0) {
        show(feedback, "Select at least one image first.", "error")
        return
    }
    const formData = new FormData()
    Array.from(input.files).forEach((file) => formData.append("files", file))
    try {
        const response = await fetch(apiUrl("/files"), { method: "POST", body: formData })
        const data = await response.json()
        if (!response.ok) throw new Error(data.detail || "Upload failed")
        show(feedback, `Uploaded ${data.length} file(s).`, "success")
        input.value = ""
        await refreshFiles()
    } catch (error) {
        show(feedback, error.message, "error")
    }
}

const handleImport = async (event) => {
    event.preventDefault()
    const feedback = document.getElementById("import-feedback")
    const sourceDir = document.getElementById("source-dir").value.trim()
    if (!sourceDir) {
        show(feedback, "Provide a directory path.", "error")
        return
    }
    try {
        const data = await jsonFetch(apiUrl("/files/import"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ source_dir: sourceDir, copy: true }),
        })
        show(feedback, `Imported ${data.imported.length} file(s), skipped ${data.skipped.length}.`, "success")
        await refreshFiles()
    } catch (error) {
        show(feedback, error.message, "error")
    }
}

const handleDeleteFile = async (event) => {
    const button = event.target.closest("[data-delete-file]")
    if (!button) return
    const filename = button.dataset.deleteFile
    if (!window.confirm(`Remove ${filename}?`)) return
    try {
        const response = await fetch(apiUrl(`/files/${encodeURIComponent(filename)}`), { method: "DELETE" })
        if (!response.ok && response.status !== 204) {
            const data = await response.json().catch(() => ({}))
            throw new Error(data.detail || "Delete failed")
        }
        await refreshFiles()
    } catch (error) {
        window.alert(error.message)
    }
}

const handleSaveDoc = async (event) => {
    const button = event.target.closest("[data-save-doc]")
    if (!button) return
    const box = button.closest(".json-box")
    const docName = box.dataset.doc
    const textarea = box.querySelector("[data-doc-textarea]")
    const feedback = box.querySelector("[data-doc-feedback]")
    const statusEl = box.querySelector("[data-doc-status]")
    const raw = textarea.value.trim()
    let payload = null
    if (raw) {
        try {
            payload = JSON.parse(raw)
        } catch (error) {
            show(feedback, `Invalid JSON: ${error.message}`, "error")
            return
        }
    }
    try {
        const response = await fetch(apiUrl(`/${docName}`), {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        })
        const data = await response.json()
        if (!response.ok) throw new Error(data.detail || "Save failed")
        show(feedback, data.status === "deleted" ? "Deleted." : "Saved.", "success")
        if (statusEl) statusEl.textContent = raw ? "present" : "empty"
    } catch (error) {
        show(feedback, error.message, "error")
    }
}

const handleClearDoc = (event) => {
    const button = event.target.closest("[data-clear-doc]")
    if (!button) return
    const textarea = button.closest(".json-box").querySelector("[data-doc-textarea]")
    textarea.value = ""
}

const handleRun = async () => {
    const errorEl = document.getElementById("last-error")
    try {
        await jsonFetch(apiUrl("/process"), { method: "POST" })
        updateStatus("queued", null)
        setTimeout(pollStatus, 1000)
    } catch (error) {
        show(errorEl, error.message, "error")
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const uploadForm = document.getElementById("upload-form")
    if (uploadForm) uploadForm.addEventListener("submit", handleUpload)
    const importForm = document.getElementById("import-form")
    if (importForm) importForm.addEventListener("submit", handleImport)
    const runBtn = document.getElementById("run-btn")
    if (runBtn) runBtn.addEventListener("click", handleRun)
    document.addEventListener("click", (event) => {
        handleDeleteFile(event)
        handleSaveDoc(event)
        handleClearDoc(event)
    })

    const initialStatus = document.getElementById("status-pill")
    if (initialStatus) {
        const current = initialStatus.textContent.trim()
        if (current === "queued" || current === "running") {
            setTimeout(pollStatus, 1000)
        }
    }
})
