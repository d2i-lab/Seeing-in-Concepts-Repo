import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { IOU } from "./pages/iou";
import { Label } from "./pages/label";

export default function SlicingApp() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="iou" element={<IOU />} />
        <Route path="label" element={<Label />} />
      </Routes>
    </BrowserRouter>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<SlicingApp />);