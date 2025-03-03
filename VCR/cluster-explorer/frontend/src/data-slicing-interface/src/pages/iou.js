import React, { useCallback, useRef, useState, useEffect, useMemo } from 'react';

import axios from 'axios';
import { AgGridReact } from 'ag-grid-react'
import Button from 'react-bootstrap/Button';
import Col from 'react-bootstrap/Col';
import Form from 'react-bootstrap/Form';
import Row from 'react-bootstrap/Row';
import Spinner from 'react-bootstrap/Spinner';

import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-alpine.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import { COLUMNS } from '../components/columns'
import { ImageModal } from '../components/imageModal';
import { MyModal } from '../components/confirm'

const base_url = process.env.REACT_APP_API_URL

function IOUPage({ reload }) {
  const [originalData, setOriginalData] = useState(() => []);
  const [data, setData] = useState(() => []);
  const [availableFiles, setAvailableFiles] = useState(() => []);
  const [labelChoices, setLabelChoices] = useState(() => []);
  const [columnDefs, _] = useState(COLUMNS);
  const [isLoading, setIsLoading] = useState(false);
  const [imageLoading, setImagLoading] = useState(false);
  const [imageData, setImageData] = useState(() => []);

  const fileName = useRef();
  const limit = useRef();
  const label = useRef();
  const count = useRef();
  const crowding = useRef();
  const bbox_area = useRef();
  const qcut = useRef();
  const cut = useRef();
  const dedup = useRef();

  const gridRef = useRef();

  const [showModal, setShowModal] = useState(false);
  const [modalText, setModalText] = useState('testing 123');
  const [modalInputs, setModalInputs] = useState([])

  const handleModalClose = () => {
    setShowModal(false);
  };

  // const handleModalSubmit = (inputValue) => {(async()=>{
  const handleModalSubmit = async (inputValue) => {
    let result = await axios.post(`${base_url}/cluster/updatefile`, {
      "file": fileName.current,
      // "data": renaming,
      "data": {},
      "pairing": inputValue,
    }, {
      headers: {
        'Content-Type': 'application/json'
      },
    })
    // If conflict, then open dialog
    console.log(result)
    if (result.status != 200) {
      alert('Unexpected error occrured')
      return;
    }
    result = result.data

    if (!result.success) {
      alert('Rename Failed: Conflict exists. Try again.')
      setModalText('A conflict between labels occurred. Please rename your labels.')
      console.log(result.conflict)
      setModalInputs(result.conflict)
      return false
    }
    await axios.get(`${base_url}/v1/reset_cache`)
    alert('Success!')
    return true
  }

  const handleModalForceSubmit = async (inputValue) => {
    console.log('Force merge')
    let result = await axios.post(`${base_url}/cluster/force-updatefile`, {
      "file": fileName.current,
      "data": {},
      "pairing": inputValue,
    }, {
      headers: {
        'Content-Type': 'application/json'
      },
    })

    await axios.get(`${base_url}/v1/reset_cache`)
    if (result.status != 200) {
      alert('Unexpected error.')
    }

    setShowModal(false)

  }


  useEffect(() => {
    (async () => {
      let result = await axios.get(`${base_url}/v1/dir`)
      setAvailableFiles(result.data)
      let choices_result = await axios.get(`${base_url}/v1/label_choices`)
      setLabelChoices(["None"].concat(choices_result.data));
    })();
  }, [reload]);

  let defaultColDef = useMemo(() => ({
    resizable: true,
    sortable: true,
    wrapHeaderText: true,
  }), [reload]);

  const [columnTypes, setColumnTypes] = useMemo(() => ([
    {
      float: {
        filter: 'agNumberColumnFilter',
        // width: 150, 
        field: 'number',
        valueFormatter: params => params.value.toFixed(4),
      },
      int: {
        filter: 'agNumberColumnFilter',
        // width: 150, 
        field: 'number',
      },
      description: {
        width: 250,
        wrapText: true,
        cellStyle: { 'white-space': 'normal' },
        autoHeight: true,
      },
    }
  ]), []);

  const onSelectionChanged = useCallback(async () => {
    const selectedRows = gridRef.current.api.getSelectedRows();
    console.log(selectedRows)
    if (selectedRows.length > 0) {
      console.log(selectedRows[0].itemsets.toString())
      console.log('Using these params', fileName.current)
      setImageData([])
      setImagLoading(true)
      let result = await axios.post(`${base_url}/v1/visualize2`, {
        "file": fileName.current,
        "slice": selectedRows[0].itemsets.toString(),
        "limit": 0,
        "label": label.current,

        // Additional column metadata options
        "dedup": dedup.current,
        "count": count.current, // "count": "True
        "crowding": crowding.current,
        "bbox_area": bbox_area.current,
        "qcut": qcut.current,
        "cut": cut.current,
      }, {
        headers: {
          'Content-Type': 'application/json'
        },
        // responseType: 'blob',
      })
      // setImageData(result.data)
      setImageData(result.data)
      setImagLoading(false)
      console.log('this is result', result.data)
    }
  }, []);

  const saveRenaming = async () => {
    let originalLabels = originalData.map((row) => row.itemsets)
    originalLabels = originalLabels.map((row) => row.map((item) => item.split("=")[0]))


    let rowData = [];
    gridRef.current.api.forEachNode(node => rowData.push(node.data));


    let newLabels = rowData.map((row) => row.itemsets.toString())
    newLabels = newLabels.map((row) => row.split(",").map((item) => item.split("=")[0]))

    let renaming = {}
    let changed = {}
    console.log("originalLabels", originalLabels)
    for (let i = 0; i < originalLabels.length; i++) {
      for (let j = 0; j < originalLabels[i].length; j++) {
        if (changed[originalLabels[i][j]]) continue
        if (originalLabels[i][j] !== newLabels[i][j]) {
          changed[originalLabels[i][j]] = true
        }
        renaming[originalLabels[i][j]] = newLabels[i][j]
      }
    }

    let result = await axios.post(`${base_url}/cluster/updatefile`, {
      "file": fileName.current,
      "data": renaming,
    }, {
      headers: {
        'Content-Type': 'application/json'
      },
    })

    await axios.get(`${base_url}/v1/reset_cache`)

    console.log(result)
    if (result.status != 200) {
      alert('Unexpected error occrured')
      return;
    }
    result = result.data

    // If conflict, then open dialog
    if (!result.success) {
      setShowModal(true)
      setModalText('A conflict between labels occurred. Please rename your labels.')
      setModalInputs(result.conflict)
    }


  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsLoading(true);
    console.log(event.target.label.value, 'this is label', typeof (event.target.label.value))
    let labelValue = labelChoices.indexOf(event.target.label.value) - 1
    // let labelValue = -1;
    // if (event.target.label.value !== '') {
    //   labelValue = labelChoices.indexOf(event.target.label.value)
    // }

    // Set limit and file name
    fileName.current = event.target.file.value
    label.current = labelValue
    count.current = event.target.count.value
    crowding.current = event.target.crowding.checked
    bbox_area.current = event.target.bbox_area.checked

    if (event.target.count.value == 'binary') {
      count.current = false
    } else if (event.target.count.value == 'count') {
      count.current = true
    }

    qcut.current = false
    cut.current = false
    //q-discretize
    if (count.current == 'q-discretize') {
      count.current = true
      qcut.current = true
    }
    if (count.current == 'discretize') {
      count.current = true
      cut.current = true
    }

    // Don't allow both qcut and cut to be true
    if (qcut.current && cut.current) {
      alert('Cannot have both qcut and cut selected')
      setIsLoading(false)
      return
    }
    console.log('qcut', qcut.current)
    console.log('cut', cut.current)
    console.log('count', count.current)

    // qcut.current = event.target.qcut.checked
    // cut.current = event.target.cut.checked

    let result2 = await axios.post(`${base_url}/v1/sync-mine`, {
      "max_combo": parseInt(event.target.max_combo.value),
      "limit": 0,
      "support": parseFloat(event.target.support.value),
      "label": labelValue,
      "top_k": parseInt(event.target.top_k.value),
      "file": event.target.file.value,

      // Additional column metadata options
      // "count": event.target.count.checked, // "count": "True
      "dedup": event.target.dedup.value,
      "count": count.current,
      "crowding": event.target.crowding.checked,
      "bbox_area": event.target.bbox_area.checked,
      // "qcut": event.target.qcut.checked,
      // "cut": event.target.cut.checked,
      "qcut": qcut.current,
      "cut": cut.current,
    }, {
      headers: {
        'Content-Type': 'application/json'
      }
    })
    setData(JSON.parse(result2.data))
    setOriginalData(JSON.parse(result2.data))
    setIsLoading(false)
  }


  return (
    <div className="container">
      {/* <div className="ag-theme-alpine" style={{height: 1200, width: 1000}}> */}
      {/* <div className="ag-theme-alpine ag-container" style={{width: 3000}}> */}
      <div class="ag-theme-alpine ag-container">
        <Form noValidate onSubmit={handleSubmit}>
          <Row className="mb-3">
            <Form.Group as={Col} controlId="label">
              <Form.Label>Label</Form.Label>
              {/* <Form.Control type="text" placeholder="[none]" /> */}
              <Form.Select as={Col}>
                {labelChoices.length === 0 ? (<option>...</option>) : (
                  // labelChoices.slice(0, 10).map((file) => (
                  ['None', 'person', 'car', 'boat', 'chair', 'book', 'bottle', 'broccoli'].map((file) => (
                    <option key={file}>{file}</option>
                  ))
                )}
              </Form.Select>
            </Form.Group>
            <Form.Group as={Col} controlId="max_combo">
              <Form.Label>Max Combo</Form.Label>
              <Form.Control type="number" defaultValue="3" />
            </Form.Group>
            <Form.Group as={Col} controlId="support">
              <Form.Label>Support</Form.Label>
              <Form.Control type="number" defaultValue="0.01" />
            </Form.Group>
          </Row>
          <Row className="mb-3">
            {/* <Form.Group as={Col} controlId="limit">
              <Form.Label>Limit</Form.Label>
              <Form.Control type="number" defaultValue="50000" />
            </Form.Group> */}
            <Form.Group as={Col} controlId="top_k">
              <Form.Label>Top K</Form.Label>
              <Form.Control type="number" defaultValue="30" />
            </Form.Group>
            <Form.Group as={Col} controlId="file">
              <Form.Label>File</Form.Label>
              <Form.Select as={Col}>
                {availableFiles.length === 0 ? (<option>...</option>) : (
                  availableFiles.map((file) => (
                    <option key={file}>{file}</option>
                  ))
                )}
              </Form.Select>
            </Form.Group>
          </Row>
          <Row className="mb-3">
            {/* <Form.Group as={Col} controlId='count'>
              <Form.Check type="checkbox" label="Count Concepts" />
            </Form.Group> */}
            <Form.Group as={Col} controlId='dedup'>
              <Form.Label>Dedup Method</Form.Label>
              <Form.Select as={Col}>
                <option value='none'>None</option>
                <option value='subsets'>Subsets (faster)</option>
                <option value='coverage'>Coverage (higher quality, slower)</option>
              </Form.Select>
            </Form.Group>
            <Form.Group as={Col} controlId='count'>
              <Form.Label>Concept Count</Form.Label>
              <Form.Select as={Col}>
                <option value='binary'>Binary</option>
                <option value='count'>Count (no bins)</option>
                <option value='q-discretize'>Count Quartile</option>
                <option value='discretize'>Count Discretize</option>
              </Form.Select>
            </Form.Group>
            <Form.Group as={Col} controlId='crowding'>
              {/* <Form.Label>Crowding</Form.Label> */}
              {/* <Form.Control type="number"/> */}
              <Form.Check type="checkbox" label="Crowding" />
            </Form.Group>
            <Form.Group as={Col} controlId='bbox_area'>
              {/* <Form.Label>BBox Area</Form.Label> */}
              {/* <Form.Control type="number"/> */}
              <Form.Check type="checkbox" label="BBox-Area" />
            </Form.Group>
            {/* <Form.Group as={Col} controlId='qcut'> */}
            {/* <Form.Label>BBox Area</Form.Label> */}
            {/* <Form.Control type="number"/> */}
            {/* <Form.Check type="checkbox" label="q-Discretize" /> */}
            {/* </Form.Group> */}
            {/* <Form.Group as={Col} controlId='cut'> */}
            {/* <Form.Label>BBox Area</Form.Label> */}
            {/* <Form.Control type="number"/> */}
            {/* <Form.Check type="checkbox" label="Discretize" /> */}
            {/* </Form.Group> */}

          </Row>
          <Button variant="primary" type="submit" className="mb-3">
            Submit
          </Button> &nbsp;
          <Button variant="primary" type="button" className="mb-3" onClick={() => saveRenaming()}>
            Save Renaming
          </Button>
        </Form>
        {isLoading ? (<div>Loading</div>) :
          <div class="h-2/4">
            <AgGridReact
              ref={gridRef}
              rowData={data}
              columnDefs={columnDefs}
              defaultColDef={defaultColDef}
              onGridReady={() => { gridRef.current.api.sizeColumnsToFit(); }}
              columnTypes={columnTypes}
              onSelectionChanged={onSelectionChanged}
              rowSelection={'single'}
            // sideBar={'filters'}
            />

          </div>
        }
      </div>
      <div class="image-container overflow-auto ml-10">
        {imageLoading.valueOf() ? (<Spinner />) : (
          imageData.map((image) => (
            <ImageModal src={`${base_url}/v1/img/${image}`} />
          ))
        )}
      </div>
      <div>
        <MyModal show={showModal} handleClose={handleModalClose} submitCall={handleModalSubmit} forceSubmit={handleModalForceSubmit} text={modalText} inputs={modalInputs}>
        </MyModal>
      </div>
    </div>
  )
}

export const IOU = IOUPage;
