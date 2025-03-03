import './App.css';
import ClusteringApp from './clustering-interface/src/ClusteringApp';
import { IOU } from "./data-slicing-interface/src/pages/iou"
import { Navbar, Nav } from 'react-bootstrap';
import React, { useState } from 'react';

// require('dotenv').config()
// console.log(process.env)

function App() {
  const [selectedPage, setSelectedPage] = useState("iou");
  const [reload, setReload] = useState(true)
  const [debounce, setDebounce] = useState(false)


  const handlePageChange = (page) => {
    setSelectedPage(page);
    if (page == "iou") {
      setReload(true)
    } else {
      setReload(false)
      if (!debounce) setDebounce(true)
    }
  };

  return (
    <>
      <Navbar bg="light" expand="lg">
        <Navbar.Toggle aria-controls="basic-navbar-nav" />
        <Navbar.Collapse id="basic-navbar-nav">
          <Nav className="mr-auto">
            <div className="flex content-center">


              <div class={`${reload ? "border-b-4 border-gray-500" : ""}`}>
                <Nav.Link className="border" onClick={() => handlePageChange("iou")}>Miner</Nav.Link>
              </div>
              <div class={`${reload ? "" : "border-b-4 border-gray-500"}`}>
                <Nav.Link className="border" onClick={() => handlePageChange("clustering")}>Concept Explorer</Nav.Link>
              </div>
            </div>

          </Nav>
        </Navbar.Collapse>
      </Navbar>

      <div style={{ display: selectedPage === "iou" ? 'block' : 'none' }}>
        <IOU reload={reload} />
      </div>
      <div style={{ display: selectedPage === "clustering" ? 'block' : 'none' }}>
        <ClusteringApp reload={reload} debounce={debounce} />
      </div>
    </>
  );
}

export default App;
