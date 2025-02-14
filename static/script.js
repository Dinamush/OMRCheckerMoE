document.getElementById('downloadForm').addEventListener('submit', function(event) {
    const baseUrl = document.getElementById('base_url').value;
    const favoritesUrl = document.getElementById('favorites_url').value;

    if (!baseUrl.startsWith('https://') || !favoritesUrl.startsWith('https://')) {
        alert('Please enter valid URLs starting with "https://".');
        event.preventDefault();
    }
});
