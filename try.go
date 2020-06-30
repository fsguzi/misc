package main 

import (
	"fmt"
	"os"
	"django-go/cmd/utils"
	"django-go/pkg/loader"
	"fsguzi/problem"
)

func main() {

	directories := utils.AdjustDirectorys(os.Args[1:])

	for _, dir := range directories {

		directory := dir

		dataLoader := loader.NewLoader(directory)
		
		rule, _ := dataLoader.LoadRule() 
		nodes, _ := dataLoader.LoadNodes() 
		apps, _ := dataLoader.LoadApps()
		
		_, _, _, MAXINSTANCE, NODETYPES := problem.ScheduleInit(nodes, apps, rule)

		fmt.Println(MAXINSTANCE)
		fmt.Println(NODETYPES)
	}
}

