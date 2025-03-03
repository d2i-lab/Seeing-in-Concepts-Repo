import React, { useState } from 'react';
import Modal from 'react-bootstrap/Modal';
import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';

export const MyModal = ({ show, handleClose, inputs, submitCall, text, forceSubmit }) => {
  const [inputValues, setInputValues] = useState(inputs.map(input => input[1]));


  const handleInputChange = (index, value) => {
    const newInputValues = [...inputValues];
    newInputValues[index] = value;
    setInputValues(newInputValues);
  };

  const handleSubmit = async () => {
    // Handle submission logic here
    console.log('Submitted:', inputValues);
    // let success = submitCall(inputValues)
    let new_inputs = {}
    inputValues.map((val, index) => {
      if (inputs[index]) new_inputs[inputs[index][0]] = val
    }

    ) // might not work

    if (Object.values(new_inputs).length == 0) {
      alert('Cannot use empty input!')
      return
    }

    for (const [k, v] of Object.entries(new_inputs)) {
      if (!v) {
        alert('Cannot use empty input!')
        return
      }
    }
    let success = await submitCall(new_inputs)

    // Reset input values after submission if needed
    if (success) {
      setInputValues(Array(inputs.length).fill(''));
      handleClose();
    }
    else {
      console.log('hey still stay here')
      show = true
    }
  };

  const handleForceSubmit = () => {
    console.log('Submitted:', inputValues);

    let new_inputs = {}
    inputValues.map((val, index) => {

      if (inputs[index]) new_inputs[inputs[index][0]] = val
    })

    if (Object.values(new_inputs).length == 0) {
      alert('Cannot use empty input!')
      return
    }

    for (const [k, v] of Object.entries(new_inputs)) {
      if (!v) {
        alert('Cannot use empty input!')
        return
      }
    }
    let success = forceSubmit(new_inputs)

    // Reset input values after submission if needed
    if (success) {
      setInputValues(Array(inputs.length).fill(''));
      forceSubmit();
    }
  }

  return (
    <Modal show={show} onHide={handleClose}>
      <Modal.Header closeButton>
        <Modal.Title> {text} </Modal.Title>
      </Modal.Header>
      <Modal.Body>
        {inputs.map((input, index) => (
          <>
            <Form.Group>
              <Form.Label>{"Original Label:"}</Form.Label>
              <Form.Control
                type="text"
                value={input[0]}
                readOnly
                disabled
              />
            </Form.Group>
            <Form.Group key={index}>
              <Form.Label>{"Propose New Label: "}</Form.Label>
              <Form.Control
                type="text"
                onChange={(e) => handleInputChange(index, e.target.value)}
                placeholder={input[1] + ' (Suggested)'}
              ></Form.Control>
            </Form.Group>
          </>
        ))}
      </Modal.Body>
      <Modal.Footer>
        <Button variant="secondary" onClick={handleClose}>
          Cancel
        </Button>
        <Button variant="primary" onClick={handleSubmit}>
          Propose New Label (no merge)
        </Button>
        <Button variant="primary" onClick={handleForceSubmit}>
          Force-Merge
        </Button>
      </Modal.Footer>
    </Modal>
  );
};

