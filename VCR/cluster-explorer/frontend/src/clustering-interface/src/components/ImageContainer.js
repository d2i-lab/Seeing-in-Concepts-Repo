import React, { useState, useEffect } from 'react';

const base_url = process.env.REACT_APP_API_URL
let current_cluster_id = null
let imageIndex = null
let segmentIndex = null
let imageList = null
const ImageContainer = ({ data, selectedClusterInformation, setSelectedClusterInformation }) => {
    const [imageData, setImageData] = useState(null)

    useEffect(() => {
        async function temp() {
            if (selectedClusterInformation != null) {
                let cluster_id = selectedClusterInformation["cluster_id"]
                imageList = Object.keys(data["cluster_map"][cluster_id])
                let image_name = selectedClusterInformation["img_name"]
                let segment_id = selectedClusterInformation["segment_id"]
                let imageUrl = await fetchSingleImage(image_name, segment_id)
                console.log(imageUrl)
                // setImageData({ "main": imageUrl, "clusterName": selectedClusterInformation["cluster_name"], "imageCount": imageList.length })

                setImageData(previousImageData => {
                    let newImageData = { ...previousImageData }
                    newImageData["main"] = imageUrl
                    newImageData["clusterName"] = selectedClusterInformation["cluster_name"]
                    newImageData["imageCount"] = imageList.length
                    return newImageData
                })


                if (current_cluster_id != cluster_id) {
                    current_cluster_id = cluster_id
                    imageIndex = 0
                    segmentIndex = -1
                    imageList = Object.keys(data["cluster_map"][cluster_id])
                    getNextImage(cluster_id)
                }
            }
        }
        temp()
    }, [selectedClusterInformation])



    const fetchSingleImage = async (image_name, segment_id) => {
        try {
            const response = await fetch(`${base_url}/cluster/images/?image_name=${image_name}&segment_id=${segment_id}`)
            let imageUrl
            if (response.ok) {
                imageUrl = URL.createObjectURL(await response.blob());;
            } else return null
            return imageUrl
        }
        catch {
            return null
        }
    }


    const getPreviousImage = async () => {
        if (imageIndex >= 0) {
            segmentIndex = segmentIndex - 1
            let image_name = imageList[imageIndex]

            if (segmentIndex < 0) {
                if (imageIndex == 0) imageIndex = imageList.length - 1
                else imageIndex = Math.max(imageIndex - 1, 0)
                image_name = imageList[imageIndex]
                segmentIndex = data["cluster_map"][selectedClusterInformation["cluster_id"]][image_name].length - 1
            }

            let segment_id = data["cluster_map"][selectedClusterInformation["cluster_id"]][image_name][segmentIndex]
            let imageUrl = await fetchSingleImage(image_name, segment_id)
            setImageData(previousImageData => {
                let newImageData = { ...previousImageData }
                newImageData["additional"] = imageUrl
                return newImageData
            })
        }
    }

    const getNextImage = async () => {
        if (imageIndex == imageList.length - 1) {
            imageIndex = 0
            segmentIndex = -1
        }


        segmentIndex = segmentIndex + 1
        let image_name = imageList[imageIndex]

        if (segmentIndex > data["cluster_map"][selectedClusterInformation["cluster_id"]][image_name].length - 1) {
            segmentIndex = 0
            imageIndex = imageIndex + 1
            image_name = imageList[imageIndex]
        }

        let segment_id = data["cluster_map"][selectedClusterInformation["cluster_id"]][image_name][segmentIndex]
        let imageUrl = await fetchSingleImage(image_name, segment_id)
        if (imageUrl != null) {
            setImageData(previousImageData => {
                let newImageData = { ...previousImageData }
                newImageData["additional"] = imageUrl
                return newImageData
            })
        }
        else console.log("BRUH")
    }



    return (
        <>
            {imageData && <div class="flex flex-col items-center border w-full min-h-[800px] mr-10 text-center">
                {/* <div class="flex justify-center items-center"> */}
                Cluster "{imageData["clusterName"]}" with {imageData["imageCount"]} segments
                <br></br>
                Selected Image
                <img src={imageData["main"]} class="max-h-[325px] w-auto"></img>
                {/* </div> */}


                <div class="flex justify-center items-center gap-2 mt-2 mb-2">

                    <button class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
                        onClick={() => getPreviousImage()}>
                        Previous
                    </button>

                    {/* Additional Images from Cluster {clusterNames[selectedClusterInformation["cluster_id"]]} with {imageList.length} segments */}
                    Additional Images

                    <button class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
                        onClick={() => getNextImage()}>
                        Next
                    </button>
                </div>

                <img src={imageData["additional"]} class="max-h-[325px] w-auto"></img>
            </div>}

        </>
    )
}

export default ImageContainer