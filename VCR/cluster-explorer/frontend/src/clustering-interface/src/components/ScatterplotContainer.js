import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';

let imageIndex = null
let segmentIndex = null
let imageList = null
let cluster_id = null
// TODO: going to have to make a map from cluster label to the og indices of clusters labels contained in that cluster label now after merges
// TODO: have to also figure out how to deal with indices after splitting clusters

const ScatterplotContainer = ({ data, setSelectedClusterInformation }) => {
    let centroids = data["centroids"]
    let nearest_points = data["nearest_points"]
    let nearest_points_indices = data["nearest_points_indices"]
    let indexMap = data["index_map"]

    const [refresh, setRefresh] = useState(false)

    const handleClick = async (event) => {
        if (event.points && event.points.length > 0) {
            const clickedPoint = event.points[0];
            const metadata = clickedPoint.customdata;

            if (!metadata) return

            let img_name = metadata["img_name"]
            let segment_id = metadata["segment_id"]
            let clicked_cluster_id = metadata["cluster_label"]

            setSelectedClusterInformation(previousSelectedClusterInformation => {
                let newSelectedClusterInformation = { ...previousSelectedClusterInformation }
                newSelectedClusterInformation["cluster_id"] = clicked_cluster_id
                newSelectedClusterInformation["cluster_name"] = data["cluster_names"][clicked_cluster_id]
                newSelectedClusterInformation["img_name"] = img_name
                newSelectedClusterInformation["segment_id"] = segment_id
                return newSelectedClusterInformation
            })


        }

    };


    const colors = centroids.map((_, index) => `hsl(${(360 / centroids.length) * index}, 100%, 50%)`);


    let trace = {
        x: nearest_points.flatMap(nearest_point => nearest_point.map(point => point[0])), // x-axis
        y: nearest_points.flatMap(nearest_point => nearest_point.map(point => point[1])), // y-axis
        text: nearest_points_indices.flatMap(nearest_points_idx => nearest_points_idx.map(point => data["cluster_names"][indexMap[point]["cluster_label"]])),
        type: 'scatter',
        mode: 'markers',
        hoverinfo: 'text',
        customdata: nearest_points_indices.flatMap(nearest_points_idx => nearest_points_idx.map(point => indexMap[point])),
        marker: {
            color: nearest_points_indices.flatMap(nearest_points_idx => nearest_points_idx.map(point => colors[indexMap[point]["cluster_label"]]))
        }
    }

    return (
        <>
            <div class="flex flex-row gap-6">
                <div class="border-[6px] border-gray-500">
                    <div onClick={() => setRefresh(true)}>

                        <Plot data={[trace]}
                            onClick={handleClick}
                            layout={{
                                plot_bgcolor: 'white',
                                showlegend: false,
                                xaxis: {
                                    uirevision: 'time',
                                    visible: false,
                                    // fixedrange: true // should fix clicking 
                                },
                                yaxis: {
                                    uirevision: 'time',
                                    visible: false,
                                    // fixedrange: true // should fix clicking 
                                },
                                width: 700,
                                height: 400,
                                margin: { t: 0, l: 0, r: 0, b: 0 },
                            }} />
                    </div>
                </div>
            </div>

        </>

    )

}



export default ScatterplotContainer