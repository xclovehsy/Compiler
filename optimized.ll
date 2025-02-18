; ModuleID = '/tmp/tmpi1soabqn/optimized.bc'
source_filename = "/root/Compiler/data/anghabench/qsort.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: nounwind uwtable
define dso_local i32 @partition(i32* noundef %0, i32 noundef %1, i32 noundef %2) #0 {
  %4 = sext i32 %2 to i64
  %5 = getelementptr inbounds i32, i32* %0, i64 %4
  %6 = load i32, i32* %5, align 4, !tbaa !5
  %7 = add nsw i32 %1, -1
  br label %8

8:                                                ; preds = %41, %3
  %.0 = phi i32 [ %7, %3 ], [ %.1, %41 ]
  %storemerge = phi i32 [ %1, %3 ], [ %42, %41 ]
  %9 = icmp slt i32 %storemerge, %2
  br i1 %9, label %24, label %10

10:                                               ; preds = %8
  %11 = add nsw i32 %.0, 1
  %12 = sext i32 %11 to i64
  %13 = getelementptr inbounds i32, i32* %0, i64 %12
  %14 = load i32, i32* %13, align 4, !tbaa !5
  %15 = sext i32 %2 to i64
  %16 = getelementptr inbounds i32, i32* %0, i64 %15
  %17 = load i32, i32* %16, align 4, !tbaa !5
  %18 = add nsw i32 %.0, 1
  %19 = sext i32 %18 to i64
  %20 = getelementptr inbounds i32, i32* %0, i64 %19
  store i32 %17, i32* %20, align 4, !tbaa !5
  %21 = sext i32 %2 to i64
  %22 = getelementptr inbounds i32, i32* %0, i64 %21
  store i32 %14, i32* %22, align 4, !tbaa !5
  %23 = add nsw i32 %.0, 1
  ret i32 %23

24:                                               ; preds = %8
  %25 = sext i32 %storemerge to i64
  %26 = getelementptr inbounds i32, i32* %0, i64 %25
  %27 = load i32, i32* %26, align 4, !tbaa !5
  %28 = icmp slt i32 %27, %6
  br i1 %28, label %29, label %41

29:                                               ; preds = %24
  %30 = add nsw i32 %.0, 1
  %31 = sext i32 %30 to i64
  %32 = getelementptr inbounds i32, i32* %0, i64 %31
  %33 = load i32, i32* %32, align 4, !tbaa !5
  %34 = sext i32 %storemerge to i64
  %35 = getelementptr inbounds i32, i32* %0, i64 %34
  %36 = load i32, i32* %35, align 4, !tbaa !5
  %37 = sext i32 %30 to i64
  %38 = getelementptr inbounds i32, i32* %0, i64 %37
  store i32 %36, i32* %38, align 4, !tbaa !5
  %39 = sext i32 %storemerge to i64
  %40 = getelementptr inbounds i32, i32* %0, i64 %39
  store i32 %33, i32* %40, align 4, !tbaa !5
  br label %41

41:                                               ; preds = %24, %29
  %.1 = phi i32 [ %30, %29 ], [ %.0, %24 ]
  %42 = add nsw i32 %storemerge, 1
  br label %8, !llvm.loop !9
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nounwind uwtable
define dso_local void @quickSort(i32* noundef %0, i32 noundef %1, i32 noundef %2) #0 {
  %4 = icmp slt i32 %1, %2
  br i1 %4, label %5, label %9

5:                                                ; preds = %3
  %6 = call i32 @partition(i32* noundef %0, i32 noundef %1, i32 noundef %2)
  %7 = add nsw i32 %6, -1
  call void @quickSort(i32* noundef %0, i32 noundef %1, i32 noundef %7)
  %8 = add nsw i32 %6, 1
  call void @quickSort(i32* noundef %0, i32 noundef %8, i32 noundef %2)
  br label %9

9:                                                ; preds = %5, %3
  ret void
}

attributes #0 = { nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { argmemonly nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 1}
!4 = !{!"Ubuntu clang version 14.0.0-1ubuntu1.1"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = distinct !{!9, !10, !11}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!"llvm.loop.unroll.disable"}