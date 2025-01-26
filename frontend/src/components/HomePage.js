import React, { useState } from 'react';
import { Button, CircularProgress } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import '../styles/HomePage.css';

function HomePage({ onUpload }) {
    const [image, setImage] = useState(null);
    const [loading, setLoading] = useState(false);
    const [sketch, setSketch] = useState(null);

    const handleImageUpload = (e) => {
        const file = e.target.files[0];
        if (file) {
            setImage({
                file,
                url: URL.createObjectURL(file),
                name: file.name,
            });
        }
    };

    const handleSubmit = async () => {
        if (!image) {
            alert('Please upload an image first!');
            return;
        }
        setLoading(true);

        const formData = new FormData();
        formData.append('file', image.file);

        setTimeout(async () => {
            try {
                const response = await fetch('http://localhost:5000/generate-sketch', {
                    method: 'POST',
                    body: formData,
                });
                const blob = await response.blob();
                setSketch(URL.createObjectURL(blob));
            } catch (error) {
                alert('Error generating sketch: ' + error.message);
            } finally {
                setLoading(false);
            }
        }, 2000);
    };

    const handleDownload = () => {
        const link = document.createElement('a');
        link.href = sketch;
        link.download = 'sketch.png';
        link.click();
    };

    const handleReset = () => {
        setImage(null);
        setSketch(null);
    };

    return (
        <div className="container">
            <img src="/logo.PNG" alt="Logo" className="logo" />

            {!loading && !sketch && (
                <>
                    <h2>Upload the picture you'd like to turn into a sketch</h2>
                    <div className="upload-box">
                        <label htmlFor="file-upload" className="upload-label">
                            <CloudUploadIcon fontSize="large" />
                            <p>Browse and choose the files you want to upload from your computer</p>
                        </label>
                        <input
                            id="file-upload"
                            type="file"
                            style={{ display: 'none' }}
                            onChange={handleImageUpload}
                        />
                        {image && (
                            <div className="file-preview">
                                <img src={image.url} alt="Uploaded" className="small-preview-image" />
                                <span className="file-name">{image.name}</span>
                            </div>
                        )}
                    </div>
                    <div className="button-container-single">
                        <Button
                            variant="contained"
                            className="sketch-btn"
                            onClick={handleSubmit}
                            disabled={!image}
                        >
                            Sketch!
                        </Button>
                    </div>
                </>
            )}

            {loading && (
                <div className="loading-container">
                    <CircularProgress size={70} style={{ color: 'black' }} />
                    <p className="loading-text">Drawing your sketch</p>
                </div>
            )}

            {sketch && (
                <div className="result-container">
                    <div className="image-row">
                        <img src={image.url} alt="Original" className="image-box" />
                        <img src={sketch} alt="Sketch" className="image-box" />
                    </div>
                    <div className="button-container">
                        <Button
                            variant="contained"
                            className="download-btn"
                            onClick={handleDownload}
                        >
                            Download your sketch
                        </Button>
                        <Button
                            variant="outlined"
                            className="sketch-again-btn"
                            onClick={handleReset}
                        >
                            Sketch again
                        </Button>
                    </div>
                </div>
            )}

            <img src="/pencil.png" alt="Pencil Icon" className="pencil-img"/>

        </div>
    );
}

export default HomePage;
