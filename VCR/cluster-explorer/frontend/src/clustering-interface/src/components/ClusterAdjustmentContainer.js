import React, { useState } from 'react';
import { AutoComplete } from 'primereact/autocomplete';

const ClusterAdjustmentContainer = ({ data, selectedClusterInformation, setSelectedClusterInformation }) => {
    const [searchValue, setSearchValue] = useState("")
    const [suggestions, setSuggestions] = useState(Object.values(data["cluster_names"]))
    const [renameClusterName, setRenameClusterName] = useState("")

    const searchClusterNames = (event) => {
        // filter suggestions based on query
        if (event.query.length === 0) {
            return data["cluster_names"]
        }
        let filteredSuggestions = Object.values(data["cluster_names"]).filter((suggestion) => {
            return suggestion.toLowerCase().startsWith(event.query.toLowerCase());
        });
        setSuggestions(filteredSuggestions)
    }

    const searchValues = () => {
        // search indexmap values for the cluster label

        for (const [key, value] of Object.entries(data["index_map"])) {
            let cluster_id = value["cluster_label"]
            if (data["cluster_names"][cluster_id] === searchValue) {
                // handleClick({ points: [{ customdata: value }] })

                let img_name = value["img_name"]
                let segment_id = value["segment_id"]
                let clicked_cluster_id = value["cluster_label"]

                setSelectedClusterInformation(previousSelectedClusterInformation => {
                    let newSelectedClusterInformation = { ...previousSelectedClusterInformation }
                    newSelectedClusterInformation["cluster_id"] = clicked_cluster_id
                    newSelectedClusterInformation["cluster_name"] = data['cluster_names'][clicked_cluster_id]
                    newSelectedClusterInformation["img_name"] = img_name
                    newSelectedClusterInformation["segment_id"] = segment_id
                    return newSelectedClusterInformation
                })

                break
            }
        }
    }

    const renameCluster = () => {
        if (renameClusterName != '') {

            // TODO: Not sure if this is best way to send updates:
            // data is a prop that is passed from App.js to Scatterplot.js, so
            // we're updating via reference.
            let cluster_id = selectedClusterInformation["cluster_id"]
            data['cluster_names'][cluster_id] = renameClusterName

            setSelectedClusterInformation(previousSelectedClusterInformation => {
                let newSelectedClusterInformation = { ...previousSelectedClusterInformation }
                newSelectedClusterInformation["cluster_name"] = renameClusterName
                return newSelectedClusterInformation
            })
        }

    }

    return (
        <>
            <div class="text-center">
                <div class="flex flex-row">
                    <div id="ClusterAdjustmentContainer" class="px-2 flex flex-col items-center">

                        <div class="flex items-center px-2 pb-8">
                            Search Cluster:
                            <div class="border-2 border-black mr-2" >

                                <AutoComplete value={searchValue} suggestions={suggestions} completeMethod={(e) => searchClusterNames(e)} onChange={(e) => setSearchValue(e.value)} />
                            </div>
                            <button class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
                                onClick={() => searchValues()}>
                                Search
                            </button>
                        </div>

                        <div class="flex items-center px-2 pb-8">
                            Rename Cluster: <input type="text" class="border-2 border-black mr-2" onChange={(event) => { setRenameClusterName(event.target.value) }} />
                            <button class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
                                onClick={(event) => renameCluster(event.target.value)}>
                                Rename
                            </button>
                        </div>
                    </div>
                </div>
            </div >

        </>
    )
}

export default ClusterAdjustmentContainer