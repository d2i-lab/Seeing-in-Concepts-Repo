import React, { useCallback, useRef, useState, useEffect, useMemo } from 'react';

import axios from 'axios';
import { AgGridReact } from 'ag-grid-react'
import Button from 'react-bootstrap/Button';
import Col from 'react-bootstrap/Col';
import Form from 'react-bootstrap/Form';
import Row from 'react-bootstrap/Row';

import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-alpine.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import { COLUMNS_LABEL } from '../components/columns_label'
import { ImageModal } from '../components/imageModal';

const base_url = process.env.REACT_APP_API_URL

function LabelPage() {
  const [data, setData] = useState(() => []);
  const [availableFiles, setAvailableFiles] = useState(() => []);
  const [columnDefs, _] = useState(COLUMNS_LABEL);
  const [isLoading, setIsLoading] = useState(false);
  const [imageData, setImageData] = useState(() => []);

  const fileName = useRef();
  const limit = useRef();
  const label = useRef();

  const gridRef = useRef();

  useEffect(() => {
    (async () => {
      let result = await axios.get(`${base_url}/label/dir`)
      setAvailableFiles(result.data)
    })();
  }, []);

  let defaultColDef = useMemo(() => ({
    // flex: 1,
    resizable: true,
    sortable: true,
    wrapHeaderText: true,
    wrapText: true,
    width: 125,
  }), []);

  const [columnTypes, setColumnTypes] = useMemo(() => ([
    {
      float: {
        filter: 'agNumberColumnFilter',
        field: 'number',
        valueFormatter: params => params.value.toFixed(4),
      },
      int: {
        filter: 'agNumberColumnFilter',
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
      console.log('Using these params', fileName.current, limit.current)
      setImageData([])
      let result = await axios.post(`${base_url}/label/visualize2`, {
        "file": fileName.current,
        "slice": selectedRows[0].itemsets.toString(),
        "limit": limit.current,
        "label": label.current,
      }, {
        headers: {
          'Content-Type': 'application/json'
        },
        // responseType: 'blob',
      })
      // setImageData(result.data)
      setImageData(result.data)
      console.log('this is result', result.data)
    }
  }, []);


  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsLoading(true);
    console.log(event.target.label.value, 'this is label', typeof (event.target.label.value))
    let labelValue = -1;
    if (event.target.label.value !== '') {
      labelValue = parseInt(event.target.label.value)
    }

    // Set limit and file name
    limit.current = parseInt(event.target.limit.value)
    fileName.current = event.target.file.value
    label.current = labelValue

    // let result2 = await axios.post('http://localhost:4443/label/sync-mine', {
    let result2 = await axios.post(`${base_url}/label/sync-mine`, {
      "max_combo": parseInt(event.target.max_combo.value),
      "limit": parseInt(event.target.limit.value),
      "support": parseFloat(event.target.support.value),
      "label": labelValue,
      "top_k": parseInt(event.target.top_k.value),
      "file": event.target.file.value,
    }, {
      headers: {
        'Content-Type': 'application/json'
      }
    })
    setData(JSON.parse(result2.data))
    setIsLoading(false)
  }


  return (
    <div className="container">
      {/* <div className="ag-theme-alpine" style={{height: 1200, width: 1000}}> */}
      {/* <div className="ag-theme-alpine ag-container" style={{width: 3000}}> */}
      <div className="ag-theme-alpine ag-container" style={{ width: '100%' }}>
        <Form noValidate onSubmit={handleSubmit}>
          <Row className="mb-3">
            <Form.Group as={Col} controlId="label">
              <Form.Label>Label</Form.Label>
              <Form.Control type="text" placeholder="[none]" />
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
            <Form.Group as={Col} controlId="limit">
              <Form.Label>Limit</Form.Label>
              <Form.Control type="number" defaultValue="20000" />
            </Form.Group>
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
          <Button variant="primary" type="submit" className="mb-3">
            Submit
          </Button>
        </Form>
        {isLoading ? (<div>Loading</div>) : (
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
          />)
        }
      </div>
      <div className="image-container">
        {imageData.length === 0 ? (<div>...</div>) : (
          imageData.map((image) => (
            <ImageModal src={`${base_url}/label/img/${image}`} />
          ))
        )}
      </div>
    </div>
  )
}

export const Label = LabelPage;
