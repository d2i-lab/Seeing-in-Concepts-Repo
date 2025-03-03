import React, { useState, useEffect } from 'react';
import Modal from './Modal';
import axios from 'axios';


const Options = ({ ncentroids, setNcentroids, dimensionality, setDimensionality, embeddingsType, setEmbeddingsType, onClusterClick, exportModalOpen, setExportModalOpen, setOutputName, importModalOpen, setImportModalOpen, setInputName, reload }) => {
    // const [ncentroids, setNcentroids] = useState(532)
    // const [dimensionality, setDimensionality] = useState(2)
    // const [reduceDims, setReduceDims] = useState(false);

    const [availableFiles, setAvailableFiles] = useState(() => []);
    const base_url = process.env.REACT_APP_API_URL

    useEffect(() => {
        (async () => {
            let result = await axios.get(`${base_url}/v1/dir`)
            setAvailableFiles(result.data)
        })();
    }, [reload]);


    const onSubmit = (outputName) => {
        setOutputName(outputName);
    }

    return (
        <>
            <form class="flex flex-row justify-center" onSubmit={(event) => { event.preventDefault(); onClusterClick(); }}>


                <div class="flex flex-row p-2 max-w-xs whitespace-pre">
                    Number of Centroids:&nbsp;&nbsp;&nbsp;
                    <div class="flex gap-2">
                        <input type="number" min="0" value={ncentroids} onChange={(event) => { setNcentroids(event.target.value) }} class="border-2 border-black w-[75px]" />
                    </div>
                </div>

                <div class="hidden flex flex-col p-2 max-w-xs whitespace-pre">
                    Dimensionality for Clustering:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                    <div class="flex gap-2">
                        <input type="number" min="0" max="512" value={isNaN(dimensionality) ? "512" : dimensionality} onChange={(event) => { event.target.value === '512' ? setDimensionality("No Reduction") : setDimensionality(event.target.value) }} class="border-2 border-black" />
                        <div>{isNaN(dimensionality) ? dimensionality : ""}</div>
                    </div>
                </div>

                <div class="hidden flex flex-col p-2 max-w-xs">
                    Embeddings:
                    <div class="flex gap-2">
                        MaskCLIP
                        <button type="button" class={`w-12 h-6 rounded-full bg-blue-500 focus:outline-none`} onClick={() => { setEmbeddingsType(prevType => (prevType == "MaskCLIP" ? "Dino" : "MaskCLIP")) }}>
                            <div
                                class={`w-6 h-6 rounded-full transform transition ${embeddingsType == "Dino" ? 'translate-x-6' : ''} bg-white shadow-md`}>
                            </div>
                        </button>
                        Dino
                    </div>
                </div>

                <input type="submit" class="bg-blue-500 hover:bg-blue-700 text-white font-bold px-4 rounded" value="Run Clustering" />

                <button type="button" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 ml-2 rounded"
                    onClick={() => setExportModalOpen(true)}>
                    Export
                </button>

                <button type="button" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 ml-2 rounded"
                    onClick={() => setImportModalOpen(true)}>
                    Import Labels
                </button>
            </form>
            <Modal isModalOpen={exportModalOpen} setModalOpen={setExportModalOpen} onSubmit={onSubmit} modalType="Export" />
            <Modal isModalOpen={importModalOpen} setModalOpen={setImportModalOpen} onSubmit={(inputName) => setInputName(inputName)} modalType="Import" choices={availableFiles} />
        </>
    )
}

export default Options