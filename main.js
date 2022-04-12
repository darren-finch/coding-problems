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