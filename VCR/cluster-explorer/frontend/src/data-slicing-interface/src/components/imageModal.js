import {useState} from 'react'
import Modal from 'react-bootstrap/Modal';
import Form from 'react-bootstrap/Form';

import '../index.css';

export const ImageModal = ({title, src}) => {
    const [isOpen, setOpen] = useState(false)
    const [isFullScreen, setFullScreen] = useState(false)
    const handleClick = (open) => {
        setOpen(open)
        if (!open) {
            setFullScreen(false)
        }
    }

    const handleFormChange = (event) => {
        if (event.target.id == "fullscreen") {
            setFullScreen(event.target.checked)
        }
    }

    return (
        <div className='img-container'>
            <img className='img' src={src} onClick={()=>handleClick(true)}/>
            {isOpen && (
                <Modal 
                show={isOpen}
                onHide={()=>handleClick(false)}
                fullscreen={isFullScreen}
                id='main-modal'
                dialogClassName='modal-75w'
                >
                    <Modal.Header closeButton>
                        <Modal.Title>{title}</Modal.Title>
                    </Modal.Header>
                    <Modal.Body>
                        <img class='img' src={src}/>
                    </Modal.Body>
                    <Form onChange={handleFormChange} className='mb-2'>
                        <Form.Switch type='switch' label='Fullscreen' id='fullscreen'/>
                    </Form>
                </Modal>
            )}
        </div>
    )

}

