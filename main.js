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
