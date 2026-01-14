convert the natural language content to fit the dataset JSON structure. 
Return only the JSON structure result, no explanation
dataset {
	type: "order" or "goods" or "sales", 
	date: {
		eq : "yyyy-MM-dd'T'HH:mm:ss", // date
		gte: "yyyy-MM-dd'T'HH:mm:ss", // (after, start, over) date
		lte: "yyyy-MM-dd'T'HH:mm:ss"  // (before, end, under) date
	},
	price : {
		eq : 0,  // price
		gte : 0, // (start, over, high, up, expensive, better, premium, luxury) price
		lte : 0  // (end, under, low, down, cheap, discount) price
	},
	quantity : {
		eq : 0,  // quantity
		gte : 0, // (start, over, high, large, many, up) quantity
		lte : 0  // (end, under, low, small, little, few, down) quantity
	}
}