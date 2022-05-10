function doubleChar(str) {
	var p1 = 0;
	var result = "";
	while (p1 < str.length) {
		result += str[p1];
		result += str[p1];
		p1++;
	}
	return result;
}

function betterThanAverage(classPoints, yourPoints) {
	return yourPoints > classPoints.reduce((a, b) => a + b, 0) / classPoints.length; 
  }

const rps = (p1, p2) => {
	if(p1 == p2)
		return "Draw!";
	else {
		if((p1=="rock"&&p2=="scissors")||(p1=="scissors"&&p2=="paper")||(p1=="paper"&&p2=="rock"))
			return "Player 1 won!";
		else 
			return "Player 2 won!";
	}
};



function plural(n) {
	return true
  }