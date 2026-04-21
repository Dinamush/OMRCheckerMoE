const showFeedback = (element, message, kind = "info") => {
    if (!element) return
    element.hidden = false
    element.textContent = message
    element.classList.remove("error", "success")
    if (kind === "error") element.classList.add("error")
    if (kind === "success") element.classList.add("success")
}

const postJson = async (url, body) => {
    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    })
    const text = await response.text()
    const data = text ? JSON.parse(text) : {}
    if (!response.ok) {
        const detail = data.detail || `Request failed (${response.status})`
        throw new Error(detail)
    }
    return data
}

const deleteJson = async (url) => {
    const response = await fetch(url, { method: "DELETE" })
    if (!response.ok && response.status !== 204) {
        const data = await response.json().catch(() => ({}))
        throw new Error(data.detail || `Request failed (${response.status})`)
    }
}

const handleCreateBatch = async (event) => {
    event.preventDefault()
    const form = event.currentTarget
    const feedback = document.getElementById("create-feedback")
    const input = form.querySelector("#batch-name")
    const name = input.value.trim()
    if (!name) return

    try {
        const batch = await postJson("/api/v1/batches", { name })
        showFeedback(feedback, `Created ${batch.name}. Redirecting...`, "success")
        window.location.href = `/batches/${batch.id}`
    } catch (error) {
        showFeedback(feedback, error.message, "error")
    }
}

const handleDeleteBatch = async (event) => {
    const button = event.target.closest("[data-delete-batch]")
    if (!button) return
    const batchId = button.dataset.deleteBatch
    if (!window.confirm("Delete this batch and all its files?")) return
    try {
        await deleteJson(`/api/v1/batches/${batchId}`)
        window.location.reload()
    } catch (error) {
        window.alert(error.message)
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const createForm = document.getElementById("create-batch-form")
    if (createForm) createForm.addEventListener("submit", handleCreateBatch)
    document.addEventListener("click", handleDeleteBatch)
})
