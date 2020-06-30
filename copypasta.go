package problem

import (
	"django-go/pkg/types"
	"django-go/pkg/constants"
)

type String string

func (s String) ValueWithDefault(def string) string {
	if len(s) == 0 {
		return def
	}
	return string(s)
}

type StringSlice []string


func toResourceScoreMap(rule types.Rule) map[types.Resource]map[string]int {

	resourceMap := make(map[types.Resource]map[string]int)

	for _, rw := range rule.NodeResourceWeights {

		mmn := String(rw.NodeModelName).ValueWithDefault(constants.SCORE_EMPTY_NODE_MODEL_NAME)

		if _, ok := resourceMap[rw.Resource]; ok {
			resourceMap[rw.Resource][mmn] = rw.Weight
		} else {
			resourceMap[rw.Resource] = map[string]int{mmn: rw.Weight}
		}
	}

	return resourceMap
}


func resourceScore(scoreMap map[types.Resource]map[string]int, node types.Node) int {

	sumScore := 0

	for _, r := range types.AllResources {

		wm, ok := scoreMap[r]

		if !ok {
			wm = make(map[string]int)
		}

		if w, ok := wm[node.NodeModelName]; ok {
			sumScore += w * node.Value(r)
			continue
		}

		if w, ok := wm[constants.SCORE_EMPTY_NODE_MODEL_NAME]; ok {
			sumScore += w * node.Value(r)
		}

	}

	return sumScore
}
