import React, { useState } from 'react';
import HomePage from './components/HomePage';

function App() {
    const [image, setImage] = useState(null);

    const handleImageUpload = (event) => {
        setImage(event.target.files[0]);
    };

    return <HomePage onImageUpload={handleImageUpload} />;
}

export default App;
