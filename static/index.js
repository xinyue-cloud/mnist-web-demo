(function () {

  let canvas = document.getElementById("canvas")
  let info = document.getElementById("info")
  let ctx = canvas.getContext("2d")

  canvas.width = 280
  canvas.height = 280

  let mouse = {x: 0, y: 0}
  let last_mouse = {x: 0, y: 0}

  ctx.fillStyle = "black"
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  ctx.color = "white"
  ctx.lineWidth = 12
  ctx.lineJoin = ctx.lineCap = "round"

  canvas.addEventListener("mousemove", function (e) {
    last_mouse.x = mouse.x
    last_mouse.y = mouse.y
    info.textContent = 'x: ' + e.layerX + ', y: ' + e.layerX
    mouse.x = e.layerX - 8;
    mouse.y = e.layerY - 8;
  }, false)

  let onPaint = function () {
    //ctx.lineWidth = ctx.lineWidth;
    ctx.lineJoin = "round"
    ctx.lineCap = "round"
    ctx.strokeStyle = ctx.color
    ctx.beginPath()
    ctx.moveTo(last_mouse.x, last_mouse.y)
    ctx.lineTo(mouse.x, mouse.y)
    ctx.closePath()
    ctx.stroke()
  }

  canvas.addEventListener('mousedown', function (e) {
    canvas.addEventListener('mousemove', onPaint, false)
  }, false)


  canvas.addEventListener('mouseup', function (e) {
    canvas.removeEventListener('mousemove', onPaint, false)
  }, false)

  let makePredict = function (data) {
    let text = '';
    for (let i in data) {
      text += i + ' = ' + data[i] + '<br />'
    }
    $('#prediction').html(text);
  }

  $("#btnPredict").on('click', function () {
    let root_path = window.appConfig.rootPath;
    let canvas_obj = document.getElementById("canvas");
    let img = canvas_obj.toDataURL();
    $('.predict-result').text('...');
    $.ajax({
      type: "POST",
      url: root_path + "/predict",
      data: img,
      success: function (data) {
        $('.predict-result').text(data.label);
        makePredict(data.prediction);
      }
    });
  });

  $("#btnClear").on("click", function () {
    ctx.clearRect(0, 0, 280, 280);
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    $('.predict-result').text('...');
    $('#prediction').html('');
  });


  $("#btnTestImg").on("click", function () {
    let root_path = window.appConfig.rootPath;
    $.ajax({
      url: root_path + '/test-image',
      method: 'GET',
      xhrFields: {
        responseType: 'blob'
      },
      success: function (response) {
        // 创建图像元素并加载数据
        let img = new Image();
        img.onload = function () {
          // 将图像绘制到Canvas上
          ctx.drawImage(img, 0, 0);
        };
        img.src = window.URL.createObjectURL(response);

        $('.predict-result').text('...');
        setTimeout(() => {
          $("#btnPredict").click()
        }, 500)
      }
    });
  });

}())