function processFile() {
    const fileInput = document.getElementById('csvFile');
    const file = fileInput.files[0];
    if (!file) {
        alert('Please select a CSV file');
        return;
    }
    const formData = new FormData();
    formData.append('file', file);
    axios.post('/process', formData, {
        headers: {
            'Content-Type': 'multipart/form-data'
        }
    })
    .then(response => {
        displayResults(response.data);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '<h2>Results:</h2>';
    resultsDiv.innerHTML += `<p>Shapes found: ${data.shapes_found.join(', ')}</p>`;
    resultsDiv.innerHTML += `<p>Reflections: ${data.reflections.join(', ')}</p>`;

    const mergedShapesDiv = document.getElementById('mergedShapes');
    mergedShapesDiv.innerHTML = '<h2>Merged Shapes:</h2>';
    mergedShapesDiv.innerHTML += data.svg;

    document.getElementById('downloadSvg').style.display = 'block';
}

function downloadSvg() {
    const svgContent = document.getElementById('mergedShapes').innerHTML;
    const blob = new Blob([svgContent], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'merged_shapes.svg';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}