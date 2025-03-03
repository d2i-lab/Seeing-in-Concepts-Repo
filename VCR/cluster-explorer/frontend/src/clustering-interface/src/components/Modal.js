import React, { useState } from 'react';
import Dropdown from 'react-bootstrap/Dropdown';


const Modal = ({ isModalOpen, setModalOpen, onSubmit, modalType, choices }) => {
    const [outputName, setOutputName] = useState('');
    const [inputName, setInputName] = useState('');

    const handleInputChange = (event) => {
        setOutputName(event.target.value);
    }

    const handleCloseModal = () => {
        setModalOpen(false);
    }

    const handleSubmitModal = () => {
        if (modalType == "Export") {
            if (outputName === '') {
                alert('Please enter a name for the output.');
                return;
            }
            console.log(`Output Name submitted: ${outputName}`);
            onSubmit(outputName);
        } else {
            if (inputName === '') {
                alert('Please select an input file.');
                return;
            }
            console.log(`Input Name submitted: ${inputName}`);
            onSubmit(inputName);
        }
        // handleCloseModal();
        // You can perform additional actions with the submitted output name.
    };

    return (
        <div class={`modal fixed inset-0 bg-gray-900 bg-opacity-50 overflow-y-auto h-screen z-50`}
            style={{ display: isModalOpen ? 'block' : 'none' }}>
            <div class="modal-content bg-white w-1/3 mx-auto my-10 p-6 rounded-lg">
                <span class="close cursor-pointer absolute top-0 right-0 text-black text-3xl" onClick={handleCloseModal}>&times;&nbsp;&nbsp;</span>
                <h2 class="text-2xl font-bold mb-4">Enter {modalType == "Export" ? "Output" : "Input"} Name</h2>
                {modalType == "Export" ? <input
                    type="text"
                    placeholder="Output Name"
                    value={outputName}
                    onChange={handleInputChange}
                    class="w-full p-2 border border-gray-300 rounded mb-4"
                /> :

                    <Dropdown className="w-full p-2 bg-blue-500 border border-gray-300 rounded mb-4 flex justify-center">
                        <Dropdown.Toggle className="bg-blue-500 border rounded" variant="success" id="dropdown-basic">

                            Select Import File
                        </Dropdown.Toggle>

                        <Dropdown.Menu>
                            {choices.map((choice, index) => (
                                <Dropdown.Item key={index} onClick={() => setInputName(choice)}>
                                    {choice}
                                </Dropdown.Item>
                            ))}
                        </Dropdown.Menu>
                    </Dropdown>
                }
                <button
                    onClick={handleSubmitModal}
                    class="bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-700 focus:outline-none focus:shadow-outline-blue active:bg-blue-800"
                >
                    Submit
                </button>
            </div>
        </div>
    );

};

export default Modal;