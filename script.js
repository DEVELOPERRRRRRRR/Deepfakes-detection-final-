const input1 = document.getElementById("fileInput");
const name1 = document.getElementById("filename1");

input1.addEventListener("change", function () {
    name1.textContent = input1.files[0].name;
});

const input2 = document.getElementById("fileInput1");
const name2 = document.getElementById("filename2");

input2.addEventListener("change", function () {
    name2.textContent = input2.files[0].name;
});
