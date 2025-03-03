import React, { useState, useEffect } from 'react';
import ScatterplotContainer from './components/ScatterplotContainer';
import Options from './components/Options';
import axios from 'axios';
import ImageContainer from './components/ImageContainer';
import ClusterAdjustmentContainer from './components/ClusterAdjustmentContainer';


// TODO: This should be passed in as an argument
const hard_coded_segment_bbox_csv = process.env.REACT_APP_BASE_CSV
const base_url = process.env.REACT_APP_API_URL

console.log(`THIS IS BASE URL ${base_url}`)

let loaded_data = false

const ClusteringApp = ({ reload, debounce }) => {
    const [data, setData] = useState(null)

    const [ncentroids, setNcentroids] = useState(400)
    const [dimensionality, setDimensionality] = useState("No Reduction")
    const [embeddingsType, setEmbeddingsType] = useState("MaskCLIP")
    const [loading, setLoading] = useState(true)
    const [importedLabels, setImportedLabels] = useState([])
    const [exportModalOpen, setExportModalOpen] = useState(false);
    const [importModalOpen, setImportModalOpen] = useState(false);

    const [imageData, setImageData] = useState(null)

    const [selectedClusterInformation, setSelectedClusterInformation] = useState(null)



    // TODO: change from useEffect to instead listen on the reload prop
    useEffect(() => {
        if (debounce && !loaded_data) {

            fetchData()
            loaded_data = true
        }
    }, [debounce])

    // UseEffect to update the labels when the imported labels change
    useEffect(() => {
        if (importedLabels == null || importedLabels.length == 0) return

        for (const [key, value] of Object.entries(data["cluster_names"])) {
            data["cluster_names"][key] = importedLabels[key]
        }
    }, [importedLabels]);


    const setOutputName = (outputName) => {
        alert(`Output Name submitted: ${outputName}`);
        let exportDict = {
            metadata: {
                fname: outputName,
                embeddingsType: embeddingsType,
                // Keep track of the input metadata
                'img_dir': data['input_metadata']['img_dir'],
                'sam_jsons_dir': data['input_metadata']['sam_jsons_dir'],
                'seg_embeddings_dir': data['input_metadata']['seg_embeddings_dir'],
                'segment_bbox_path': hard_coded_segment_bbox_csv,
            },
            cluster_map: data["cluster_map"],
            cluster_names: data["cluster_names"],
        }

        axios.post(`${base_url}/cluster/export`, {
            exportDict
        }).then(resp => {
            alert('Success!')
        }).catch((err) => {
            alert(err)
        }).finally(() => {
            setExportModalOpen(false);
        })
    }

    const setInputName = (inputName) => {
        alert(`Input Name submitted: ${inputName}`);

        axios.post(`${base_url}/cluster/import`, {
            "file": inputName
        }).then(resp => {
            setImportedLabels(resp.data['labels'])
            alert('Success!')
        }).catch((err) => {
            alert(err)
        }).finally(() => {
            setExportModalOpen(false);
        })
    }

    const fetchData = async () => {
        setLoading(true)
        const dimVal = isNaN(dimensionality) ? -1 : dimensionality
        const response = await fetch(`${base_url}/cluster/generatecluster/?ncentroids=${ncentroids}&dimensionality=${dimVal}&embeddingsType=${embeddingsType}`)
        if (response.ok) {
            const data = await response.json()
            setLoading(false)
            setData(data)
        }
    }

    return (
        <>
            <div class="flex flex-row gap-10 justify-items-start mt-2">
                <div class="flex flex-col gap-10 justify-items-start ml-10">
                    <div class="flex flex-row gap-10 justify-items-start mt-2">
                        <Options ncentroids={ncentroids} setNcentroids={setNcentroids}
                            dimensionality={dimensionality} setDimensionality={setDimensionality}
                            embeddingsType={embeddingsType} setEmbeddingsType={setEmbeddingsType}
                            onClusterClick={fetchData}
                            exportModalOpen={exportModalOpen} setExportModalOpen={setExportModalOpen} setOutputName={setOutputName}
                            importModalOpen={importModalOpen} setImportModalOpen={setImportModalOpen} setInputName={setInputName}
                            reload={reload} />
                    </div>
                    {loading ?
                        <div className="flex flex-col justify-center items-center w-[700px] h-[400px] border-[6px] border-gray-500">
                            <div className="animate-spin rounded-full border-4 border-indigo-200 border-l-transparent h-20 w-20 mb-4"></div>
                            <p>Loading Clusters...</p>
                        </div> :
                        <>
                            <ScatterplotContainer data={data} setSelectedClusterInformation={setSelectedClusterInformation} />
                            <ClusterAdjustmentContainer data={data} selectedClusterInformation={selectedClusterInformation} setSelectedClusterInformation={setSelectedClusterInformation} />
                        </>
                    }
                </div>
                {data && <ImageContainer data={data} selectedClusterInformation={selectedClusterInformation} setSelectedClusterInformation={setSelectedClusterInformation} />}
            </div>


        </>
    )

}


export default ClusteringApp