// src/ShapeDetection.js
import React, { useState } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';

const ShapeDetection = () => {
    const [file, setFile] = useState(null);
    const [choice, setChoice] = useState('1');
    const [images, setImages] = useState([]);
    const [svgUrl, setSvgUrl] = useState('');

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
    };

    const handleChoiceChange = (event) => {
        setChoice(event.target.value);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();

        if (!file) {
            alert('Please upload a file.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        formData.append('choice', choice);

        try {
            const response = await fetch('http://210.18.155.129:5000/upload', { // Updated IP address and port
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.images) {
                setImages(data.images);
            }

            if (data.svg_url) {
                setSvgUrl(data.svg_url);
            }
        } catch (error) {
            console.error('Error:', error);
        }
    };

    return (
        <div className="container" style={{ padding: '20px', maxWidth: '900px' }}>
            <h1>Shape Detection and Processing</h1>
            <form onSubmit={handleSubmit}>
                <div className="form-group">
                    <label htmlFor="file">Upload CSV File:</label>
                    <input
                        type="file"
                        className="form-control-file"
                        id="file"
                        onChange={handleFileChange}
                        required
                    />
                </div>
                <div className="form-group">
                    <label htmlFor="choice">Select Action:</label>
                    <select
                        className="form-control"
                        id="choice"
                        value={choice}
                        onChange={handleChoiceChange}
                        required
                    >
                        <option value="1">Detect Shape</option>
                        <option value="2">Detect Symmetry</option>
                        <option value="3">Complete Curves</option>
                    </select>
                </div>
                <div className="button-group">
                    <button type="submit" className="btn btn-primary">Process</button>
                    {svgUrl && (
                        <div id="downloadBtn" style={{ display: 'inline-block', marginLeft: '10px' }}>
                            <a id="downloadLink" className="btn btn-success" href={svgUrl} download>Download SVG</a>
                        </div>
                    )}
                </div>
            </form>
            <div className="img-container" style={{ marginTop: '20px' }}>
                {images.map((imgData, index) => (
                    <img key={index} src={imgData} className="img-fluid" alt={`Result ${index}`} />
                ))}
            </div>
        </div>
    );
};

export default ShapeDetection;
