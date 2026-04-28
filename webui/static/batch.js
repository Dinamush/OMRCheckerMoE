const panel = document.querySelector("[data-batch-id]")
const batchId = panel ? panel.dataset.batchId : null
const STATUS_CLASSES = ["status-created", "status-queued", "status-running", "status-done", "status-failed", "status-cancelled"]

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

const updateStatus = (status, lastError, preprocessFailures = []) => {
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
    const warnEl = document.getElementById("preprocess-warning")
    if (warnEl) {
        if (Array.isArray(preprocessFailures) && preprocessFailures.length > 0) {
            warnEl.hidden = false
            warnEl.textContent = `Preprocessor could not locate markers in ${preprocessFailures.length} file(s): ${preprocessFailures.join(", ")}`
        } else {
            warnEl.hidden = true
            warnEl.textContent = ""
        }
    }
    const runBtn = document.getElementById("run-btn")
    if (runBtn) {
        runBtn.disabled = status === "queued" || status === "running"
        runBtn.textContent = status === "running" ? "Running..." : "Run OMR"
    }
    const stopBtn = document.getElementById("stop-btn")
    if (stopBtn) {
        stopBtn.disabled = !(status === "queued" || status === "running")
    }
    const restartBtn = document.getElementById("restart-btn")
    if (restartBtn) {
        restartBtn.disabled = status === "queued" || status === "running"
    }
}

const pollStatus = async () => {
    try {
        const data = await jsonFetch(apiUrl("/status"))
        updateStatus(data.status, data.last_error, data.preprocess_failures)
        if (data.status === "queued" || data.status === "running") {
            setTimeout(pollStatus, 2000)
        } else if (data.status === "done") {
            await refreshResults()
        } else if (data.status === "cancelled") {
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
            const previewUrl = `${apiUrl(`/files/${encodeURIComponent(file.name)}/preview`)}`
            li.innerHTML = `
                <a class="sheet-preview-link" target="_blank" rel="noopener">
                    <img class="sheet-preview-image" loading="lazy" alt="">
                </a>
                <div class="file-meta">
                    <span class="mono"></span>
                    <span class="muted small">${sizeKb} KB</span>
                    <button class="btn danger small" type="button" data-delete-file=""></button>
                </div>
            `
            const previewLink = li.querySelector(".sheet-preview-link")
            const previewImg = li.querySelector(".sheet-preview-image")
            previewLink.href = previewUrl
            previewImg.src = previewUrl
            previewImg.alt = `Preview of ${file.name}`
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
        renderResults(container, data)
    } catch (error) {
        console.error(error)
    }
}

const getResponseColumns = (columns) => {
    return (columns || []).filter((col) => !["file_id", "input_path", "output_path", "score"].includes(col))
}

const getResultCell = (row, col) => {
    if (col === "file_id") return row.file_id || ""
    if (col === "input_path") return row.input_path || ""
    if (col === "output_path") return row.output_path || ""
    if (col === "score") return row.score || ""
    return (row.responses && row.responses[col]) || ""
}

const getErasureRisk = (column, value) => {
    const answer = String(value || "").trim().toUpperCase()
    if (!/^q\d+/i.test(column) || answer.length <= 1) return 0
    const markedOptions = new Set(answer.replace(/[^A-Z]/g, "").split(""))
    if (markedOptions.size <= 1) return 0
    return Math.min(95, 65 + markedOptions.size * 10)
}

const renderResultCard = (row, responseColumns, index) => {
    const fileId = row.file_id || `File ${index + 1}`
    const answers = responseColumns.map((col) => {
        const value = getResultCell(row, col) || "—"
        const erasureRisk = getErasureRisk(col, value)
        return `
            <div class="answer-cell${erasureRisk ? " answer-cell-risk" : ""}">
                <span class="answer-key">${escapeHtml(col)}</span>
                <strong>${escapeHtml(value)}</strong>
                ${erasureRisk ? `<span class="answer-risk" title="Multiple marks detected in CSV output. This can happen when an erased option remains dark enough to be read.">Erasure risk ${erasureRisk}%</span>` : ""}
            </div>
        `
    }).join("")
    return `
        <article class="result-card" data-result-card="${index}" ${index === 0 ? "" : "hidden"}>
            <div class="flex-between result-card-head">
                <div>
                    <h3>${escapeHtml(fileId)}</h3>
                    ${row.input_path ? `<p class="muted small mono">${escapeHtml(row.input_path)}</p>` : ""}
                </div>
                ${row.score ? `<span class="pill status-done">Score ${escapeHtml(row.score)}</span>` : ""}
            </div>
            <div class="answer-grid">${answers}</div>
            <p class="muted small result-risk-note">Erasure risk is a review heuristic from the CSV output. Multiple marks on one question can indicate a student erased one answer and selected another, but the residual mark still read as filled.</p>
            ${row.output_path ? `<p class="muted small mono result-output-path">Output: ${escapeHtml(row.output_path)}</p>` : ""}
        </article>
    `
}

const renderResults = (container, data) => {
    if (!data.rows || data.rows.length === 0) {
        container.innerHTML = '<p class="muted" id="no-results-msg">No results yet. Run the batch to generate results.</p>'
        return
    }

    const columns = data.columns || []
    const responseColumns = getResponseColumns(columns)
    const tabs = data.rows.map((row, index) => {
        const fileId = row.file_id || `File ${index + 1}`
        return `
            <button
                class="result-file-tab${index === 0 ? " active" : ""}"
                type="button"
                role="tab"
                aria-selected="${index === 0 ? "true" : "false"}"
                data-result-select="${index}">
                <span class="mono">${escapeHtml(fileId)}</span>
                ${row.score ? `<span class="muted small">Score: ${escapeHtml(row.score)}</span>` : ""}
            </button>
        `
    }).join("")
    const cards = data.rows.map((row, index) => renderResultCard(row, responseColumns, index)).join("")
    const header = columns.map((col) => `<th>${escapeHtml(col)}</th>`).join("")
    const rows = data.rows.map((row) => {
        const cells = columns.map((col) => {
            const className = ["file_id", "input_path", "output_path"].includes(col) ? ' class="mono small"' : ""
            return `<td${className}>${escapeHtml(getResultCell(row, col))}</td>`
        }).join("")
        return `<tr>${cells}</tr>`
    }).join("")

    container.innerHTML = `
        <div class="results-shell">
            <div class="result-file-list" role="tablist" aria-label="Result files">${tabs}</div>
            <div class="result-file-detail">${cards}</div>
        </div>
        <details class="raw-results" open>
            <summary>Raw CSV table</summary>
            <div class="table-scroll">
                <table class="table results-table">
                    <thead><tr>${header}</tr></thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
        </details>
    `
}

const handleResultSelect = (event) => {
    const button = event.target.closest("[data-result-select]")
    if (!button) return
    const container = button.closest("#results-container")
    const selected = button.dataset.resultSelect
    container.querySelectorAll("[data-result-select]").forEach((tab) => {
        const isActive = tab.dataset.resultSelect === selected
        tab.classList.toggle("active", isActive)
        tab.setAttribute("aria-selected", isActive ? "true" : "false")
    })
    container.querySelectorAll("[data-result-card]").forEach((card) => {
        card.hidden = card.dataset.resultCard !== selected
    })
}

const escapeHtml = (value) => {
    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;")
}

const editorStates = new WeakMap()

const defaultDoc = (docName) => {
    if (docName === "template") {
        return {
            pageDimensions: [666, 515],
            bubbleDimensions: [10, 10],
            customLabels: {},
            outputColumns: [],
            fieldBlocks: {},
            preProcessors: [],
        }
    }
    if (docName === "config") {
        return {
            dimensions: {
                display_height: 515,
                display_width: 666,
                processing_height: 515,
                processing_width: 666,
            },
            outputs: {
                show_image_level: 0,
            },
        }
    }
    if (docName === "evaluation") {
        return {
            source_type: "custom",
            options: {
                questions_in_order: [],
                answers_in_order: [],
            },
            marking_schemes: {},
        }
    }
    return {}
}

const getEditorStorageKey = (box) => `omr-editor-mode:${batchId}:${box.dataset.doc}`

const createEl = (tag, className, text) => {
    const element = document.createElement(tag)
    if (className) element.className = className
    if (text !== undefined) element.textContent = text
    return element
}

const parseTextareaDoc = (box) => {
    const docName = box.dataset.doc
    const textarea = box.querySelector("[data-doc-textarea]")
    const raw = textarea.value.trim()
    if (!raw) return { ok: true, value: defaultDoc(docName) }
    try {
        const value = JSON.parse(raw)
        if (!value || typeof value !== "object" || Array.isArray(value)) {
            return { ok: false, error: `${docName}.json must be a JSON object` }
        }
        return { ok: true, value }
    } catch (error) {
        return { ok: false, error: `Invalid JSON: ${error.message}` }
    }
}

const getEditorState = (box) => {
    const existing = editorStates.get(box)
    if (existing) return existing
    const parsed = parseTextareaDoc(box)
    const state = {
        doc: parsed.ok ? parsed.value : defaultDoc(box.dataset.doc),
        mode: "code",
    }
    editorStates.set(box, state)
    return state
}

const syncTextareaFromState = (box) => {
    const state = getEditorState(box)
    const textarea = box.querySelector("[data-doc-textarea]")
    if (state.doc === null) {
        textarea.value = ""
        return
    }
    textarea.value = JSON.stringify(state.doc, null, 2)
}

const syncStateFromTextarea = (box) => {
    const parsed = parseTextareaDoc(box)
    if (!parsed.ok) return parsed
    const state = getEditorState(box)
    state.doc = parsed.value
    return parsed
}

const persistDynamicChange = (box) => {
    syncTextareaFromState(box)
    renderDynamicEditor(box)
}

const setEditorMode = (box, mode) => {
    const feedback = box.querySelector("[data-doc-feedback]")
    const codePanel = box.querySelector("[data-code-panel]")
    const dynamicPanel = box.querySelector("[data-dynamic-panel]")
    const state = getEditorState(box)

    if (mode === "ui") {
        const parsed = syncStateFromTextarea(box)
        if (!parsed.ok) {
            show(feedback, parsed.error, "error")
            return
        }
        renderDynamicEditor(box)
    } else {
        syncTextareaFromState(box)
    }

    state.mode = mode
    codePanel.hidden = mode !== "code"
    dynamicPanel.hidden = mode !== "ui"
    codePanel.style.display = mode === "code" ? "" : "none"
    dynamicPanel.style.display = mode === "ui" ? "" : "none"
    codePanel.setAttribute("aria-hidden", mode === "code" ? "false" : "true")
    dynamicPanel.setAttribute("aria-hidden", mode === "ui" ? "false" : "true")
    box.querySelectorAll("[data-doc-mode]").forEach((button) => {
        button.classList.toggle("active", button.dataset.docMode === mode)
    })
    window.localStorage.setItem(getEditorStorageKey(box), mode)
}

const makeSection = (title, description) => {
    const section = createEl("section", "dynamic-section")
    const heading = createEl("h4", null, title)
    section.appendChild(heading)
    if (description) section.appendChild(createEl("p", "muted small", description))
    return section
}

const makeField = (label, control, hint) => {
    const wrapper = createEl("label", "dynamic-field")
    wrapper.appendChild(createEl("span", null, label))
    wrapper.appendChild(control)
    if (hint) wrapper.appendChild(createEl("small", "muted", hint))
    return wrapper
}

const makeTextInput = (value, handleChange, placeholder = "") => {
    const input = document.createElement("input")
    input.value = value ?? ""
    input.placeholder = placeholder
    input.addEventListener("input", () => handleChange(input.value))
    return input
}

const makeNumberInput = (value, handleChange, options = {}) => {
    const wrap = createEl("div", "number-control")
    const input = document.createElement("input")
    input.type = "number"
    input.value = value ?? 0
    if (options.step !== undefined) input.step = String(options.step)
    if (options.min !== undefined) input.min = String(options.min)
    if (options.max !== undefined) input.max = String(options.max)
    const commit = (raw) => {
        const next = Number(raw)
        handleChange(Number.isNaN(next) ? 0 : next)
    }
    input.addEventListener("input", () => {
        if (range) range.value = input.value
        commit(input.value)
    })
    wrap.appendChild(input)

    let range = null
    if (options.slider) {
        range = document.createElement("input")
        range.type = "range"
        range.min = String(options.min ?? 0)
        range.max = String(options.max ?? 100)
        range.step = String(options.step ?? 1)
        range.value = input.value
        range.addEventListener("input", () => {
            input.value = range.value
            commit(range.value)
        })
        wrap.appendChild(range)
    }
    return wrap
}

const makeCheckbox = (value, handleChange) => {
    const input = document.createElement("input")
    input.type = "checkbox"
    input.checked = Boolean(value)
    input.addEventListener("change", () => handleChange(input.checked))
    return input
}

const makeSelect = (value, options, handleChange) => {
    const select = document.createElement("select")
    options.forEach((option) => {
        const item = document.createElement("option")
        item.value = option
        item.textContent = option
        select.appendChild(item)
    })
    select.value = value || options[0]
    select.addEventListener("change", () => handleChange(select.value))
    return select
}

const makeJsonTextarea = (value, handleChange) => {
    const textarea = document.createElement("textarea")
    textarea.rows = 5
    textarea.value = JSON.stringify(value, null, 2)
    textarea.addEventListener("change", () => {
        try {
            handleChange(JSON.parse(textarea.value || "null"))
            textarea.classList.remove("invalid")
        } catch (_error) {
            textarea.classList.add("invalid")
        }
    })
    return textarea
}

const makeTagEditor = (items, handleChange, placeholder = "Add item") => {
    const currentItems = Array.isArray(items) ? [...items] : []
    const wrap = createEl("div", "tag-editor")
    const list = createEl("div", "tag-list")
    const input = document.createElement("input")
    input.placeholder = placeholder

    const update = (nextItems) => {
        handleChange(nextItems)
    }

    currentItems.forEach((item, index) => {
        const chip = createEl("span", "tag-chip")
        const label = createEl("span", null, String(item))
        chip.appendChild(label)

        const up = createEl("button", null, "↑")
        up.type = "button"
        up.title = "Move left"
        up.disabled = index === 0
        up.addEventListener("click", () => {
            const next = [...currentItems]
            const moved = next.splice(index, 1)[0]
            next.splice(index - 1, 0, moved)
            update(next)
        })
        chip.appendChild(up)

        const down = createEl("button", null, "↓")
        down.type = "button"
        down.title = "Move right"
        down.disabled = index === currentItems.length - 1
        down.addEventListener("click", () => {
            const next = [...currentItems]
            const moved = next.splice(index, 1)[0]
            next.splice(index + 1, 0, moved)
            update(next)
        })
        chip.appendChild(down)

        const remove = createEl("button", null, "×")
        remove.type = "button"
        remove.title = "Remove"
        remove.addEventListener("click", () => {
            update(currentItems.filter((_entry, itemIndex) => itemIndex !== index))
        })
        chip.appendChild(remove)
        list.appendChild(chip)
    })

    input.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== ",") return
        event.preventDefault()
        const value = input.value.trim()
        if (!value) return
        input.value = ""
        update([...currentItems, value])
    })

    wrap.appendChild(list)
    wrap.appendChild(input)
    return wrap
}

const addPairControls = (section, label, values, handleChange, names = ["width", "height"]) => {
    const row = createEl("div", "dynamic-pair")
    const pair = Array.isArray(values) ? [...values] : [0, 0]
    names.forEach((name, index) => {
        row.appendChild(
            makeField(
                `${label} ${name}`,
                makeNumberInput(pair[index], (next) => {
                    const updated = [...pair]
                    updated[index] = next
                    handleChange(updated)
                }),
            )
        )
    })
    section.appendChild(row)
}

const updateObjectKey = (objectValue, oldKey, newKey) => {
    const safeKey = newKey.trim()
    if (!safeKey || safeKey === oldKey || objectValue[safeKey]) return oldKey
    const entries = Object.entries(objectValue)
    const rebuilt = {}
    entries.forEach(([key, value]) => {
        rebuilt[key === oldKey ? safeKey : key] = value
    })
    Object.keys(objectValue).forEach((key) => delete objectValue[key])
    Object.assign(objectValue, rebuilt)
    return safeKey
}

const makeCard = (title, actions = []) => {
    const card = createEl("article", "dynamic-card")
    const head = createEl("div", "dynamic-card-head")
    head.appendChild(createEl("h5", null, title))
    const actionWrap = createEl("div", "actions")
    actions.forEach((action) => actionWrap.appendChild(action))
    head.appendChild(actionWrap)
    card.appendChild(head)
    return card
}

const renderGenericObject = (container, objectValue, handleChange, skipKeys = new Set()) => {
    Object.entries(objectValue || {}).forEach(([key, value]) => {
        if (skipKeys.has(key)) return
        if (typeof value === "boolean") {
            container.appendChild(makeField(key, makeCheckbox(value, (next) => handleChange(key, next))))
            return
        }
        if (typeof value === "number") {
            container.appendChild(makeField(key, makeNumberInput(value, (next) => handleChange(key, next), { slider: true, min: 0, max: Math.max(100, value * 2 || 100) })))
            return
        }
        if (typeof value === "string") {
            container.appendChild(makeField(key, makeTextInput(value, (next) => handleChange(key, next))))
            return
        }
        if (Array.isArray(value) && value.every((entry) => typeof entry === "string" || typeof entry === "number")) {
            container.appendChild(makeField(key, makeTagEditor(value, (next) => handleChange(key, next))))
            return
        }
        container.appendChild(makeField(key, makeJsonTextarea(value, (next) => handleChange(key, next)), "Custom JSON"))
    })
}

const renderTemplateEditor = (box, root, doc) => {
    doc.pageDimensions = Array.isArray(doc.pageDimensions) ? doc.pageDimensions : [666, 515]
    doc.bubbleDimensions = Array.isArray(doc.bubbleDimensions) ? doc.bubbleDimensions : [10, 10]
    doc.outputColumns = Array.isArray(doc.outputColumns) ? doc.outputColumns : []
    doc.customLabels = doc.customLabels && typeof doc.customLabels === "object" ? doc.customLabels : {}
    doc.fieldBlocks = doc.fieldBlocks && typeof doc.fieldBlocks === "object" ? doc.fieldBlocks : {}
    doc.preProcessors = Array.isArray(doc.preProcessors) ? doc.preProcessors : []

    const basics = makeSection("Sheet Basics", "Normalized dimensions and output column order.")
    addPairControls(basics, "Page", doc.pageDimensions, (next) => {
        doc.pageDimensions = next
        persistDynamicChange(box)
    })
    addPairControls(basics, "Bubble", doc.bubbleDimensions, (next) => {
        doc.bubbleDimensions = next
        persistDynamicChange(box)
    })
    basics.appendChild(makeField("Output columns", makeTagEditor(doc.outputColumns, (next) => {
        doc.outputColumns = next
        persistDynamicChange(box)
    }, "q1..25")))
    root.appendChild(basics)

    const labels = makeSection("Custom Labels", "Grouped output columns such as candidate number fields.")
    Object.entries(doc.customLabels).forEach(([key, value]) => {
        const card = makeCard(key)
        card.appendChild(makeField("Name", makeTextInput(key, (next) => {
            updateObjectKey(doc.customLabels, key, next)
            persistDynamicChange(box)
        })))
        card.appendChild(makeField("Field tags", makeTagEditor(value, (next) => {
            doc.customLabels[key] = next
            persistDynamicChange(box)
        }, "cand1..10")))
        const remove = createEl("button", "btn danger small", "Remove")
        remove.type = "button"
        remove.addEventListener("click", () => {
            delete doc.customLabels[key]
            persistDynamicChange(box)
        })
        card.querySelector(".actions").appendChild(remove)
        labels.appendChild(card)
    })
    const addLabel = createEl("button", "btn small", "Add label group")
    addLabel.type = "button"
    addLabel.addEventListener("click", () => {
        let index = Object.keys(doc.customLabels).length + 1
        while (doc.customLabels[`CustomLabel${index}`]) index += 1
        doc.customLabels[`CustomLabel${index}`] = []
        persistDynamicChange(box)
    })
    labels.appendChild(addLabel)
    root.appendChild(labels)

    const blocks = makeSection("Field Blocks", "Bubble grids for questions, candidate numbers, and other sheet fields.")
    Object.entries(doc.fieldBlocks).forEach(([key, block]) => {
        const remove = createEl("button", "btn danger small", "Remove")
        remove.type = "button"
        remove.addEventListener("click", () => {
            delete doc.fieldBlocks[key]
            persistDynamicChange(box)
        })
        const card = makeCard(key, [remove])
        block.origin = Array.isArray(block.origin) ? block.origin : [0, 0]
        block.fieldLabels = Array.isArray(block.fieldLabels) ? block.fieldLabels : []
        card.appendChild(makeField("Block name", makeTextInput(key, (next) => {
            updateObjectKey(doc.fieldBlocks, key, next)
            persistDynamicChange(box)
        })))
        card.appendChild(makeField("Field type", makeSelect(block.fieldType || "QTYPE_MCQ4", ["QTYPE_MCQ4", "QTYPE_INT", "__CUSTOM__"], (next) => {
            block.fieldType = next
            persistDynamicChange(box)
        })))
        card.appendChild(makeField("Direction", makeSelect(block.direction || "vertical", ["vertical", "horizontal"], (next) => {
            block.direction = next
            persistDynamicChange(box)
        })))
        addPairControls(card, "Origin", block.origin, (next) => {
            block.origin = next
            persistDynamicChange(box)
        }, ["x", "y"])
        card.appendChild(makeField("Bubbles gap", makeNumberInput(block.bubblesGap ?? 0, (next) => {
            block.bubblesGap = next
            persistDynamicChange(box)
        }, { step: 0.1, slider: true, min: 0, max: 100 })))
        card.appendChild(makeField("Labels gap", makeNumberInput(block.labelsGap ?? 0, (next) => {
            block.labelsGap = next
            persistDynamicChange(box)
        }, { step: 0.1, slider: true, min: 0, max: 150 })))
        card.appendChild(makeField("Field labels", makeTagEditor(block.fieldLabels, (next) => {
            block.fieldLabels = next
            persistDynamicChange(box)
        }, "q1..5")))
        renderGenericObject(card, block, (childKey, next) => {
            block[childKey] = next
            persistDynamicChange(box)
        }, new Set(["origin", "fieldLabels", "fieldType", "direction", "bubblesGap", "labelsGap"]))
        blocks.appendChild(card)
    })
    const addBlock = createEl("button", "btn small", "Add field block")
    addBlock.type = "button"
    addBlock.addEventListener("click", () => {
        let index = Object.keys(doc.fieldBlocks).length + 1
        while (doc.fieldBlocks[`fieldBlock${index}`]) index += 1
        doc.fieldBlocks[`fieldBlock${index}`] = {
            origin: [0, 0],
            bubblesGap: 20,
            labelsGap: 40,
            fieldLabels: [],
            fieldType: "QTYPE_MCQ4",
        }
        persistDynamicChange(box)
    })
    blocks.appendChild(addBlock)
    root.appendChild(blocks)

    const processors = makeSection("Preprocessors", "Ordered image transforms before bubble reading.")
    doc.preProcessors.forEach((processor, index) => {
        processor.options = processor.options && typeof processor.options === "object" ? processor.options : {}
        const remove = createEl("button", "btn danger small", "Remove")
        remove.type = "button"
        remove.addEventListener("click", () => {
            doc.preProcessors.splice(index, 1)
            persistDynamicChange(box)
        })
        const up = createEl("button", "btn small", "Up")
        up.type = "button"
        up.disabled = index === 0
        up.addEventListener("click", () => {
            const moved = doc.preProcessors.splice(index, 1)[0]
            doc.preProcessors.splice(index - 1, 0, moved)
            persistDynamicChange(box)
        })
        const down = createEl("button", "btn small", "Down")
        down.type = "button"
        down.disabled = index === doc.preProcessors.length - 1
        down.addEventListener("click", () => {
            const moved = doc.preProcessors.splice(index, 1)[0]
            doc.preProcessors.splice(index + 1, 0, moved)
            persistDynamicChange(box)
        })
        const card = makeCard(`${index + 1}. ${processor.name || "Processor"}`, [up, down, remove])
        card.draggable = true
        card.dataset.arrayIndex = String(index)
        card.appendChild(makeField("Name", makeSelect(processor.name || "CropOnMarkers", ["CropOnMarkers", "CropPage", "FeatureBasedAlignment", "GaussianBlur", "Levels", "MedianBlur"], (next) => {
            processor.name = next
            persistDynamicChange(box)
        })))
        renderGenericObject(card, processor.options, (childKey, next) => {
            processor.options[childKey] = next
            persistDynamicChange(box)
        })
        card.appendChild(makeField("Options JSON", makeJsonTextarea(processor.options, (next) => {
            processor.options = next && typeof next === "object" && !Array.isArray(next) ? next : {}
            persistDynamicChange(box)
        }), "Use for nested arrays like markerCorners or referenceMarkerCenters."))
        processors.appendChild(card)
    })
    const addProcessor = createEl("button", "btn small", "Add preprocessor")
    addProcessor.type = "button"
    addProcessor.addEventListener("click", () => {
        doc.preProcessors.push({ name: "CropOnMarkers", options: { relativePath: "omr_marker.jpg" } })
        persistDynamicChange(box)
    })
    processors.appendChild(addProcessor)
    root.appendChild(processors)

    const handled = new Set(["pageDimensions", "bubbleDimensions", "customLabels", "outputColumns", "fieldBlocks", "preProcessors"])
    const advanced = makeSection("Advanced JSON", "Any custom keys not covered above.")
    renderGenericObject(advanced, doc, (key, next) => {
        doc[key] = next
        persistDynamicChange(box)
    }, handled)
    if (advanced.children.length > 2) root.appendChild(advanced)
}

const renderConfigEditor = (box, root, doc) => {
    doc.dimensions = doc.dimensions && typeof doc.dimensions === "object" ? doc.dimensions : {}
    doc.outputs = doc.outputs && typeof doc.outputs === "object" ? doc.outputs : {}
    doc.threshold_params = doc.threshold_params && typeof doc.threshold_params === "object" ? doc.threshold_params : {}
    doc.alignment_params = doc.alignment_params && typeof doc.alignment_params === "object" ? doc.alignment_params : {}

    const dimensions = makeSection("Dimensions")
    const dimensionKeys = ["display_height", "display_width", "processing_height", "processing_width"]
    dimensionKeys.forEach((key) => {
        dimensions.appendChild(makeField(key, makeNumberInput(doc.dimensions[key] ?? 0, (next) => {
            doc.dimensions[key] = next
            persistDynamicChange(box)
        }, { slider: true, min: 0, max: 3000 })))
    })
    root.appendChild(dimensions)

    const outputs = makeSection("Outputs")
    renderGenericObject(outputs, doc.outputs, (key, next) => {
        doc.outputs[key] = next
        persistDynamicChange(box)
    })
    root.appendChild(outputs)

    const thresholds = makeSection("Threshold Parameters")
    renderGenericObject(thresholds, doc.threshold_params, (key, next) => {
        doc.threshold_params[key] = next
        persistDynamicChange(box)
    })
    root.appendChild(thresholds)

    const alignment = makeSection("Alignment Parameters")
    renderGenericObject(alignment, doc.alignment_params, (key, next) => {
        doc.alignment_params[key] = next
        persistDynamicChange(box)
    })
    root.appendChild(alignment)

    const handled = new Set(["dimensions", "outputs", "threshold_params", "alignment_params"])
    const advanced = makeSection("Advanced JSON")
    renderGenericObject(advanced, doc, (key, next) => {
        doc[key] = next
        persistDynamicChange(box)
    }, handled)
    if (advanced.children.length > 1) root.appendChild(advanced)
}

const renderEvaluationEditor = (box, root, doc) => {
    doc.options = doc.options && typeof doc.options === "object" ? doc.options : {}
    doc.marking_schemes = doc.marking_schemes && typeof doc.marking_schemes === "object" ? doc.marking_schemes : {}

    const source = makeSection("Source")
    source.appendChild(makeField("source_type", makeSelect(doc.source_type || "custom", ["custom", "csv"], (next) => {
        doc.source_type = next
        persistDynamicChange(box)
    })))
    root.appendChild(source)

    const options = makeSection("Options")
    renderGenericObject(options, doc.options, (key, next) => {
        doc.options[key] = next
        persistDynamicChange(box)
    })
    root.appendChild(options)

    const schemes = makeSection("Marking Schemes")
    Object.entries(doc.marking_schemes).forEach(([key, scheme]) => {
        const remove = createEl("button", "btn danger small", "Remove")
        remove.type = "button"
        remove.addEventListener("click", () => {
            delete doc.marking_schemes[key]
            persistDynamicChange(box)
        })
        const card = makeCard(key, [remove])
        card.appendChild(makeField("Scheme name", makeTextInput(key, (next) => {
            updateObjectKey(doc.marking_schemes, key, next)
            persistDynamicChange(box)
        })))
        if (key === "DEFAULT") {
            renderGenericObject(card, scheme, (markKey, next) => {
                scheme[markKey] = next
                persistDynamicChange(box)
            })
            schemes.appendChild(card)
            return
        }
        scheme.marking = scheme.marking && typeof scheme.marking === "object" ? scheme.marking : {
            correct: 1,
            incorrect: 0,
            unmarked: 0,
        }
        if (scheme.questions !== undefined) {
            card.appendChild(makeField("Questions", makeTagEditor(scheme.questions || [], (next) => {
                scheme.questions = next
                persistDynamicChange(box)
            }, "q1..25")))
        }
        renderGenericObject(card, scheme.marking, (markKey, next) => {
            scheme.marking[markKey] = next
            persistDynamicChange(box)
        })
        renderGenericObject(card, scheme, (childKey, next) => {
            scheme[childKey] = next
            persistDynamicChange(box)
        }, new Set(["questions", "marking"]))
        schemes.appendChild(card)
    })
    const addScheme = createEl("button", "btn small", "Add marking scheme")
    addScheme.type = "button"
    addScheme.addEventListener("click", () => {
        let index = Object.keys(doc.marking_schemes).length + 1
        while (doc.marking_schemes[`SCHEME_${index}`]) index += 1
        doc.marking_schemes[`SCHEME_${index}`] = {
            questions: [],
            marking: {
                correct: 1,
                incorrect: 0,
                unmarked: 0,
            },
        }
        persistDynamicChange(box)
    })
    schemes.appendChild(addScheme)
    root.appendChild(schemes)

    const handled = new Set(["source_type", "options", "marking_schemes"])
    const advanced = makeSection("Advanced JSON")
    renderGenericObject(advanced, doc, (key, next) => {
        doc[key] = next
        persistDynamicChange(box)
    }, handled)
    if (advanced.children.length > 1) root.appendChild(advanced)
}

function renderDynamicEditor(box) {
    const root = box.querySelector("[data-dynamic-panel]")
    const state = getEditorState(box)
    const doc = state.doc
    root.innerHTML = ""

    if (doc === null) {
        root.appendChild(createEl("p", "muted", "Document cleared. Save to delete it, or switch back to UI mode from code to start a fresh document."))
        return
    }

    if (!doc || typeof doc !== "object" || Array.isArray(doc)) {
        root.appendChild(createEl("p", "error", "This editor only supports JSON objects."))
        return
    }

    if (box.dataset.doc === "template") {
        renderTemplateEditor(box, root, doc)
    } else if (box.dataset.doc === "config") {
        renderConfigEditor(box, root, doc)
    } else if (box.dataset.doc === "evaluation") {
        renderEvaluationEditor(box, root, doc)
    } else {
        renderGenericObject(root, doc, (key, next) => {
            doc[key] = next
            persistDynamicChange(box)
        })
    }
}

const refreshAssets = async () => {
    try {
        const assets = await jsonFetch(apiUrl("/assets"))
        const list = document.getElementById("asset-list")
        const empty = document.getElementById("no-assets-msg")
        const panel = document.getElementById("template-assets-panel")
        if (!panel) return

        if (!assets || assets.length === 0) {
            if (list) list.remove()
            if (!document.getElementById("no-assets-msg")) {
                const p = document.createElement("p")
                p.className = "muted"
                p.id = "no-assets-msg"
                p.textContent = "Your template does not reference any external assets."
                panel.appendChild(p)
            }
            return
        }

        if (empty) empty.remove()

        let target = list
        if (!target) {
            target = document.createElement("ul")
            target.className = "asset-list"
            target.id = "asset-list"
            panel.appendChild(target)
        }
        target.innerHTML = ""

        assets.forEach((asset) => {
            const li = document.createElement("li")
            li.dataset.assetName = asset.name

            if (asset.present) {
                const previewUrl = apiUrl(`/assets/${encodeURIComponent(asset.name)}/preview`)
                const preview = document.createElement("a")
                preview.className = "image-preview-link asset-preview-link"
                preview.href = previewUrl
                preview.target = "_blank"
                preview.rel = "noopener"

                const img = document.createElement("img")
                img.className = "image-preview-thumb asset-preview-thumb"
                img.src = previewUrl
                img.alt = `Preview of ${asset.name}`
                img.loading = "lazy"
                preview.appendChild(img)
                li.appendChild(preview)
            } else {
                const missing = document.createElement("span")
                missing.className = "image-preview-missing"
                missing.setAttribute("aria-hidden", "true")
                li.appendChild(missing)
            }

            const name = document.createElement("span")
            name.className = "mono"
            name.textContent = asset.name
            li.appendChild(name)

            const pill = document.createElement("span")
            pill.className = `pill status-${asset.present ? "done" : "failed"}`
            pill.textContent = asset.present ? "present" : "missing"
            li.appendChild(pill)

            const meta = document.createElement("span")
            meta.className = "muted small"
            if (asset.present && typeof asset.size_bytes === "number") {
                meta.textContent = `${(asset.size_bytes / 1024).toFixed(1)} KB`
            } else {
                meta.textContent = "required by template"
            }
            li.appendChild(meta)

            if (asset.present) {
                const btn = document.createElement("button")
                btn.className = "btn danger small"
                btn.type = "button"
                btn.dataset.deleteAsset = asset.name
                btn.textContent = "Remove"
                li.appendChild(btn)
            }

            target.appendChild(li)
        })
    } catch (error) {
        console.error(error)
    }
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
    const state = getEditorState(box)
    if (state.mode === "ui") {
        if (state.doc !== null) syncTextareaFromState(box)
    } else {
        const synced = syncStateFromTextarea(box)
        if (!synced.ok) {
            show(feedback, synced.error, "error")
            return
        }
    }
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
        if (docName === "template") {
            await refreshAssets()
        }
    } catch (error) {
        show(feedback, error.message, "error")
    }
}

const handleClearDoc = (event) => {
    const button = event.target.closest("[data-clear-doc]")
    if (!button) return
    const box = button.closest(".json-box")
    const state = getEditorState(box)
    const textarea = box.querySelector("[data-doc-textarea]")
    state.doc = null
    textarea.value = ""
    if (state.mode === "ui") renderDynamicEditor(box)
}

const handleEditorMode = (event) => {
    const button = event.target.closest("[data-doc-mode]")
    if (!button) return
    const box = button.closest(".json-box")
    setEditorMode(box, button.dataset.docMode)
}

const handleAssetUpload = async (event) => {
    event.preventDefault()
    const feedback = document.getElementById("asset-upload-feedback")
    const input = document.getElementById("asset-file-input")
    if (!input.files || input.files.length === 0) {
        show(feedback, "Select at least one asset file first.", "error")
        return
    }
    const formData = new FormData()
    Array.from(input.files).forEach((file) => formData.append("files", file))
    try {
        const response = await fetch(apiUrl("/assets"), { method: "POST", body: formData })
        const data = await response.json()
        if (!response.ok) throw new Error(data.detail || "Asset upload failed")
        show(feedback, `Uploaded ${data.length} asset(s).`, "success")
        input.value = ""
        await refreshAssets()
    } catch (error) {
        show(feedback, error.message, "error")
    }
}

const handleDeleteAsset = async (event) => {
    const button = event.target.closest("[data-delete-asset]")
    if (!button) return
    const name = button.dataset.deleteAsset
    if (!window.confirm(`Remove asset ${name}?`)) return
    try {
        const response = await fetch(apiUrl(`/assets/${encodeURIComponent(name)}`), { method: "DELETE" })
        if (!response.ok && response.status !== 204) {
            const data = await response.json().catch(() => ({}))
            throw new Error(data.detail || "Delete failed")
        }
        await refreshAssets()
    } catch (error) {
        window.alert(error.message)
    }
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

const handleStop = async () => {
    const errorEl = document.getElementById("last-error")
    try {
        const data = await jsonFetch(apiUrl("/cancel"), { method: "POST" })
        updateStatus(data.status, data.status === "running" ? "Stop requested. The current image will finish first." : "Cancelled before processing started.")
        setTimeout(pollStatus, 1000)
    } catch (error) {
        show(errorEl, error.message, "error")
    }
}

const handleRestart = async () => {
    const errorEl = document.getElementById("last-error")
    try {
        await jsonFetch(apiUrl("/restart"), { method: "POST" })
        updateStatus("queued", null)
        await refreshResults()
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
    const assetForm = document.getElementById("asset-upload-form")
    if (assetForm) assetForm.addEventListener("submit", handleAssetUpload)
    const runBtn = document.getElementById("run-btn")
    if (runBtn) runBtn.addEventListener("click", handleRun)
    const stopBtn = document.getElementById("stop-btn")
    if (stopBtn) stopBtn.addEventListener("click", handleStop)
    const restartBtn = document.getElementById("restart-btn")
    if (restartBtn) restartBtn.addEventListener("click", handleRestart)
    document.addEventListener("click", (event) => {
        handleDeleteFile(event)
        handleDeleteAsset(event)
        handleResultSelect(event)
        handleEditorMode(event)
        handleSaveDoc(event)
        handleClearDoc(event)
    })

    document.querySelectorAll(".json-box").forEach((box) => {
        const savedMode = window.localStorage.getItem(getEditorStorageKey(box))
        if (savedMode === "ui") {
            setEditorMode(box, "ui")
        }
    })

    const initialStatus = document.getElementById("status-pill")
    if (initialStatus) {
        const current = initialStatus.textContent.trim()
        if (current === "queued" || current === "running") {
            setTimeout(pollStatus, 1000)
        }
    }
})
