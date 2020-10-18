var center_x = 0;
var center_y = 0;
var center_z = 0;
var balls = new Array();
function setup() {
  // put setup code here
  createCanvas(windowWidth, windowHeight,WEBGL);
  
  for(var i=0; i<150; i++)
  {
    balls[i] = new Ball();
  }
}

function draw() {
  background(25,25,25);
  push();
    rotateX(mouseY/1000);
    push();
      rotateY(-mouseX/1000);
      for(var i=0; i<balls.length; i++)
      {
        balls[i].drawBall();
      }
    pop();
  pop();
  camera(0, 0, 700, 0, 0, 0, 0, 1, 0);
}

class Ball
{
  constructor()
  {
    this.moving_radius = random(100,800);
    this.angleY = random(0,360);
    this.angleZ = random(0,360);
    this.angleX = random(0,360);
  }


  drawBall()
  {
    push();
      rotateZ(this.angleZ);
      push();
        rotateX(this.angleX);
        push();
          rotateY(this.angleY);
          push();
            translate(this.moving_radius,0,0);
            noStroke();
            fill(255,255,255,180);
            sphere(2);
          pop();
        pop();
      pop();
     
      pop();
    this.angleY = this.angleY+0.0008;
  }
}

//Everytime will resize the canvas according to the brower window
function windowResized() 
{
  resizeCanvas(windowWidth, windowHeight);
}
