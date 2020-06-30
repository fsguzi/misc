package problem

import (
	"django-go/pkg/types"
	"django-go/pkg/util"
)

type NodeType struct {
	NodeModelName string
	Gpu           int       
	Cpu           int       
	Ram           int       
	Disk          int       
	Eni           int 
	Topologies    []types.Topology 
	Supply        int
	Cost		  int
}

func ScheduleInit(nodes []types.Node, apps []types.App, rule types.Rule) ([]types.App, []types.Node, types.Rule, []int, []NodeType){
	groupRuleAssociates := types.FromApps(apps)
	allMaxInstancePerNodeLimit := util.ToAllMaxInstancePerNodeLimit(rule, groupRuleAssociates)
	
	maxinstance := make([]int, len(allMaxInstancePerNodeLimit))
	for i,app := range apps {
		appgroup := app.Group
		maxinstance[i] = allMaxInstancePerNodeLimit[appgroup]
	}

	machinetype := make(map[string]int)
	nodetypes := make([]NodeType, 0)

	scoreMap := toResourceScoreMap(rule)
	for _,node := range nodes {
		typename := node.NodeModelName

		if _,ok := machinetype[typename] ; !ok {
			newNodeType := NodeType{
						NodeModelName:    node.NodeModelName,
						Gpu:              node.Gpu,    
						Cpu:              node.Cpu,     
						Ram:              node.Ram,       
						Disk:             node.Disk,       
						Eni:              node.Eni, 
						Supply:           0,
						Cost:             resourceScore(scoreMap, node)}

			nodetypes = append(nodetypes, newNodeType)
			machinetype[typename] = 0
		}
		machinetype[typename]++
	}

	
	for i:=0 ; i<len(nodetypes) ; i++ {
		typename := nodetypes[i].NodeModelName
		nodetypes[i].Supply = machinetype[typename]
	}

	return apps, nodes, rule, maxinstance, nodetypes
}
