import React, { useState } from 'react';
import Modal from './Modal';

const Options = ({ ncentroids, setNcentroids, dimensionality,
    setDimensionality, embeddingsType, setEmbeddingsType,
    onClusterClick, isModalOpen, setModalOpen, setOutputName }) => {
    // const [ncentroids, setNcentroids] = useState(532)
    // const [dimensionality, setDimensionality] = useState(2)
    // const [reduceDims, setReduceDims] = useState(false);

    const onSubmit = (outputName) => {
        setOutputName(outputName);
    }

    return (
        <>
            <div class="flex gap-10">
                <div class="flex flex-col p-2 max-w-xs whitespace-pre">
                    Number of Centroids:&nbsp;&nbsp;&nbsp;
                    <div class="flex gap-2">
                        <input type="range" min="0" max="532" value={ncentroids} onChange={(event) => { setNcentroids(event.target.value) }} />
                        <div>{ncentroids}</div>
                    </div>
                </div>

                <div class="flex flex-col p-2 max-w-xs whitespace-pre">
                    Dimensionality for Clustering:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                    <div class="flex gap-2">
                        <input type="range" min="0" max="512" value={isNaN(dimensionality) ? "512" : dimensionality} onChange={(event) => { event.target.value === '512' ? setDimensionality("No Reduction") : setDimensionality(event.target.value) }} />
                        <div>{dimensionality}</div>
                    </div>
                </div>

                <div class="flex flex-col p-2 max-w-xs">
                    Embeddings:
                    <div class="flex gap-2">
                        MaskCLIP
                        <button class={`w-12 h-6 rounded-full bg-blue-500 focus:outline-none`} onClick={() => { setEmbeddingsType(prevType => (prevType == "MaskCLIP" ? "Dino" : "MaskCLIP")) }}>
                            <div
                                class={`w-6 h-6 rounded-full transform transition ${embeddingsType == "Dino" ? 'translate-x-6' : ''} bg-white shadow-md`}>
                            </div>
                        </button>
                        Dino
                    </div>
                </div>

                <button class="bg-blue-500 hover:bg-blue-700 text-white font-bold px-4 rounded"
                    onClick={() => { onClusterClick() }}>
                    Run Clustering
                </button>

                <button class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
                    onClick={() => setModalOpen(true)}>
                    Export
                </button>

            </div>
            <Modal isModalOpen={isModalOpen} setModalOpen={setModalOpen} onSubmit={onSubmit} />
        </>
    )
}

export default Options