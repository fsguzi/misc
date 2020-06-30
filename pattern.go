package schedule

import (
	"strconv"
)


type Pattern struct{
	Use []int
	Cost float64
	Key string
	Nodename string
	Nodeindex int
	Deploy int
}

func PatternToKey(use []int) string {
	byteslice := make([]byte, len(use))
	for i,x := range use {
		b := strconv.Itoa(x)[0]
		byteslice[i] = b
	}
	key = string(byteslice)
	return key
}

func NewPattern(use []int, nodename string, nodeindex int) *Pattern {
	use := make([]int, len)
	new := Pattern{Use:use, Nodename:nodename, Nodeindex:nodeindex}
	return &new
}
