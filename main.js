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



function sumOfN(n) {
	var curSum = 0
	var result = []
	
	for (var i = 0; i <= Math.abs(n); i++) {
	  if (n > 0)
		curSum += i
	  else
		curSum -= i
	  result.push(curSum)
	}
	
	return result
  };



function toCamelCase(str){
	var strings = str.split('-').join('_').split('_');
	var result = "";
	for (var i = 0; i < strings.length; i++) {
		const curStr = strings[i];
		if (i > 0) {
		strings[i] = replaceAtIndex(curStr, 0, curStr[0].toUpperCase());
		}
	}
	return strings.reduce((prevVal, curVal) => prevVal + curVal, "");
}

function replaceAtIndex(str, index, replacement) {
	return str.substring(0, index) + replacement + str.substring(index + replacement.length);
}